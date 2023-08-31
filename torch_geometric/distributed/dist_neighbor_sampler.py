from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from ordered_set import OrderedSet
from torch import Tensor

from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore
from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.distributed.event_loop import (
    ConcurrentEventLoop,
    wrap_torch_future,
)
from torch_geometric.distributed.rpc import (
    RPCCallBase,
    RPCRouter,
    rpc_async,
    rpc_partition_to_workers,
    rpc_register,
    shutdown_rpc,
)
from torch_geometric.sampler import (
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NegativeSampling,
    NeighborSampler,
    NodeSamplerInput,
    SamplerOutput,
    edge_sample_async,
)
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.sampler.utils import remap_keys
from torch_geometric.typing import EdgeType, NodeType

from ..typing import Dict, EdgeType, NumNeighbors, OptTensor, Tuple
from ..utils import ensure_device


class RpcSamplingCallee(RPCCallBase):
    r""" A wrapper for rpc callee that will perform rpc sampling from
    remote processes.
    """
    def __init__(self, sampler: NeighborSampler, device: torch.device):
        super().__init__()
        self.sampler = sampler
        self.device = device

    def rpc_async(self, *args, **kwargs):
        output = self.sampler.sample_one_hop(*args, **kwargs)

        return output

    def rpc_sync(self, *args, **kwargs):
        pass


@dataclass
class NodeDict:
    def __init__(self, node_types):
        self.src: Dict[NodeType, Tensor] = {}
        self.with_dupl: Dict[NodeType, Tensor] = {}
        self.out: Union[OrderedSet[List[int]], Dict[NodeType, Tensor]] = {}

        for ntype in node_types:
            self.src.update({ntype: torch.empty(0, dtype=torch.int64)})
            self.with_dupl.update({ntype: torch.empty(0, dtype=torch.int64)})
            self.out.update({ntype: OrderedSet([])})


@dataclass
class BatchDict:
    def __init__(self, node_types, disjoint):
        self.src: Dict[NodeType, Tensor] = {}
        self.with_dupl: Dict[NodeType, Tensor] = {}
        self.out: Dict[NodeType, Tensor] = {}
        self.disjoint = disjoint

        if not self.disjoint:
            for ntype in node_types:
                self.src.update({ntype: torch.empty(0, dtype=torch.int64)})
                self.with_dupl.update(
                    {ntype: torch.empty(0, dtype=torch.int64)})
                self.out.update({ntype: torch.empty(0, dtype=torch.int64)})
        else:
            self.src = None
            self.with_dupl = None
            self.out = None


class DistNeighborSampler:
    r"""An implementation of an distributed and asynchronised neighbor sampler
    used by :class:`~torch_geometric.distributed.DistNeighborLoader`."""
    def __init__(
        self,
        current_ctx: DistContext,
        rpc_worker_names: Dict[DistRole, List[str]],
        data: Tuple[LocalGraphStore, LocalFeatureStore],
        channel: mp.Queue(),
        num_neighbors: Optional[NumNeighbors] = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        concurrency: int = 1,
        device: Optional[torch.device] = None,
        **kwargs,
    ):

        print(f"---- 555.1 -------- data={data}     ")
        self.current_ctx = current_ctx
        self.rpc_worker_names = rpc_worker_names

        self.data = data
        self.dist_graph = data[1]
        self.dist_feature = data[0]
        assert isinstance(
            self.dist_graph, LocalGraphStore
        ), "Provided data is in incorrect format: self.dist_graph must be "
        f"`LocalGraphStore`, got {type(self.dist_graph)}"
        assert isinstance(
            self.dist_feature, LocalFeatureStore
        ), "Provided data is in incorrect format: self.dist_feature must be "
        f"`LocalFeatureStore`, got {type(self.dist_feature)}"
        self.is_hetero = self.dist_graph.meta['is_hetero']

        self.num_neighbors = num_neighbors
        self.channel = channel
        self.concurrency = concurrency
        self.device = device
        self.event_loop = None
        self.replace = replace
        self.subgraph_type = subgraph_type
        self.disjoint = disjoint
        self.temporal_strategy = temporal_strategy
        self.time_attr = time_attr
        self.csc = True  # always true?
        self.with_edge_attr = self.dist_feature.has_edge_attr()

    def register_sampler_rpc(self):

        partition2workers = rpc_partition_to_workers(
            current_ctx=self.current_ctx,
            num_partitions=self.dist_graph.num_partitions,
            current_partition_idx=self.dist_graph.partition_idx)

        self.rpc_router = RPCRouter(partition2workers)
        self.dist_feature.set_rpc_router(self.rpc_router)

        print("---- 666.2 -------- register_rpc done    ")

        self._sampler = NeighborSampler(
            data=(self.dist_feature, self.dist_graph),
            num_neighbors=self.num_neighbors,
            subgraph_type=self.subgraph_type,
            replace=self.replace,
            disjoint=self.disjoint,
            temporal_strategy=self.temporal_strategy,
            time_attr=self.time_attr,
        )

        print(
            "----------- DistNeighborSampler: after NeigborSampler()  ------ "
        )

        # rpc register
        rpc_sample_callee = RpcSamplingCallee(self._sampler, self.device)
        self.rpc_sample_callee_id = rpc_register(rpc_sample_callee)

        print("---- 666.3 -------- register_rpc done    ")

    def init_event_loop(self):

        print(f"-----{repr(self)}: init_event_loop() Start  ")

        self.event_loop = ConcurrentEventLoop(self.concurrency)
        self.event_loop._loop.call_soon_threadsafe(ensure_device, self.device)
        self.event_loop.start_loop()

        print(
            f"----------- {repr(self)}: init_event_loop()   END ------------- "
        )

    # Node-based distributed sampling #########################################

    def sample_from_nodes(
            self, inputs: NodeSamplerInput,
            **kwargs) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
        inputs = NodeSamplerInput.cast(inputs)
        if self.channel is None:
            # synchronous sampling
            return self.event_loop.run_task(
                coro=self._sample_from(self.node_sample, inputs))

        # asynchronous sampling
        cb = kwargs.get('callback', None)
        self.event_loop.add_task(
            coro=self._sample_from(self.node_sample, inputs), callback=cb)
        return None

    # Edge-based distributed sampling #########################################

    def sample_from_edges(
        self,
        inputs: EdgeSamplerInput,
        neg_sampling: Optional[NegativeSampling] = None,
        **kwargs,
    ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
        if self.channel is None:
            # synchronous sampling
            return self.event_loop.run_task(coro=self._sample_from(
                edge_sample_async, inputs, self.node_sample,
                self._sampler.num_nodes, self.disjoint,
                self._sampler.node_time, neg_sampling, distributed=True))

        # asynchronous sampling
        cb = kwargs.get('callback', None)
        self.event_loop.add_task(
            coro=self._sample_from(edge_sample_async, inputs, self.node_sample,
                                   self._sampler.num_nodes, self.disjoint,
                                   self._sampler.node_time, neg_sampling,
                                   distributed=True), callback=cb)
        return None

    async def _sample_from(
            self, async_func, *args,
            **kwargs) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:

        sampler_output = await async_func(*args, **kwargs)
        res = await self._colloate_fn(sampler_output)

        if self.channel is None:
            return res
        self.channel.put(res)
        return None

    async def node_sample(
        self,
        inputs: NodeSamplerInput,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r"""Performs distributed sampling from a :class:`NodeSamplerInput`.
        TODO: add more comment"""
        seed = inputs.node.to(self.device)
        seed_time = inputs.time.to(
            self.device) if inputs.time is not None else None
        input_type = inputs.input_type
        self.input_type = input_type
        metadata = (seed, seed_time)
        batch_size = seed.numel()
        src_batch = torch.arange(batch_size) if self.disjoint else None

        if self.is_hetero:
            if input_type is None:
                raise ValueError("Input type should be defined")

            node_dict = NodeDict(self._sampler.node_types)
            batch_dict = BatchDict(self._sampler.node_types, self.disjoint)

            edge_dict: Dict[EdgeType, Tensor] = {}

            seed_time_dict: Dict[NodeType, Tensor] = {input_type: seed_time}
            num_sampled_nodes_dict: Dict[NodeType, List[int]] = {}

            sampled_nbrs_per_node_dict: Dict[EdgeType, List[int]] = {}
            num_sampled_edges_dict: Dict[EdgeType, List[int]] = {}

            for ntype in self._sampler.node_types:
                num_sampled_nodes_dict.update({ntype: [0]})

            for etype in self._sampler.edge_types:
                edge_dict.update({etype: torch.empty(0, dtype=torch.int64)})
                num_sampled_edges_dict.update({etype: []})
                sampled_nbrs_per_node_dict.update({etype: []})

            node_dict.src[input_type] = seed
            batch_dict.src[input_type] = src_batch if self.disjoint else None

            node_dict.out[input_type] = OrderedSet(
                seed.tolist()) if not self.disjoint else OrderedSet(
                    tuple(zip(src_batch.tolist(), seed.tolist())))
            num_sampled_nodes_dict[input_type].append(seed.numel())

            for i in range(self._sampler.num_hops):
                # create tasks
                task_dict = {}
                for etype in self._sampler.edge_types:
                    src = etype[0] if not self.csc else etype[2]

                    if node_dict.src[src].numel():
                        seed_time = seed_time_dict.get(
                            src, None) if seed_time_dict.get(
                                src, None) is not None else None
                        one_hop_num = self.num_neighbors[i] if isinstance(
                            self.num_neighbors,
                            List) else self.num_neighbors[etype][i]

                        task_dict[etype] = self.event_loop._loop.create_task(
                            self._sample_one_hop(node_dict.src[src],
                                                 one_hop_num, seed_time,
                                                 batch_dict.src[src], etype))

                for etype, task in task_dict.items():
                    out: HeteroSamplerOutput = await task

                    # remove duplicates
                    # TODO: find better method to remove duplicates
                    node_wo_dupl = OrderedSet(
                        (out.node
                         ).tolist()) if not self.disjoint else OrderedSet(
                             zip((out.batch).tolist(), (out.node).tolist()))
                    if len(node_wo_dupl) == 0:
                        # no neighbors were sampled
                        break
                    dst = etype[2] if not self.csc else etype[0]
                    duplicates = node_dict.out[dst].intersection(node_wo_dupl)
                    node_wo_dupl.difference_update(duplicates)
                    node_dict.src[dst] = Tensor(
                        node_wo_dupl if not self.disjoint else list(
                            zip(*node_wo_dupl))[1]).type(torch.int64)
                    node_dict.out[dst].update(node_wo_dupl)

                    node_dict.with_dupl[dst] = torch.cat(
                        [node_dict.with_dupl[dst], out.node])
                    edge_dict[etype] = torch.cat([edge_dict[etype], out.edge])

                    if self.disjoint:
                        batch_dict.src[dst] = Tensor(
                            list(zip(*node_wo_dupl))[0]).type(torch.int64)
                        batch_dict.with_dupl[dst] = torch.cat(
                            [batch_dict.with_dupl[dst], out.batch])

                    num_sampled_nodes_dict[dst].append(len(node_dict.src[dst]))
                    num_sampled_edges_dict[etype].append(len(out.node))
                    sampled_nbrs_per_node_dict[etype] += out.metadata

            sampled_nbrs_per_node_dict = remap_keys(sampled_nbrs_per_node_dict,
                                                    self._sampler.to_rel_type)

            row_dict, col_dict = torch.ops.pyg.hetero_relabel_neighborhood(
                self._sampler.node_types, self._sampler.edge_types,
                {input_type: seed}, node_dict.with_dupl,
                sampled_nbrs_per_node_dict, self._sampler.num_nodes,
                batch_dict.with_dupl, self.csc, self.disjoint)

            node_dict.out = {
                ntype: Tensor(node_dict.out[ntype]).type(torch.int64)
                for ntype in self._sampler.node_types
            }
            if self.disjoint:
                for ntype in self._sampler.node_types:
                    batch_dict.out[ntype], node_dict.out[
                        ntype] = node_dict.out[ntype].t().contiguous()

            sample_output = HeteroSamplerOutput(
                node=node_dict.out, row=remap_keys(row_dict,
                                                   self._sampler.to_edge_type),
                col=remap_keys(col_dict,
                               self._sampler.to_edge_type), edge=edge_dict,
                batch=batch_dict.out if self.disjoint else None,
                num_sampled_nodes=num_sampled_nodes_dict,
                num_sampled_edges=num_sampled_edges_dict, metadata=metadata)
        else:

            src = seed

            node = OrderedSet(
                src.tolist()) if not self.disjoint else OrderedSet(
                    tuple(zip(src_batch.tolist(), src.tolist())))
            node_with_dupl = []
            batch_with_dupl = []
            edge = []

            sampled_nbrs_per_node = []
            num_sampled_nodes = [seed.numel()]
            num_sampled_edges = [0]

            for one_hop_num in self.num_neighbors:
                out = await self._sample_one_hop(src, one_hop_num, seed_time,
                                                 src_batch)

                # remove duplicates
                # TODO: find better method to remove duplicates
                node_wo_dupl = OrderedSet(
                    (out.node).tolist()) if not self.disjoint else OrderedSet(
                        zip((out.batch).tolist(), (out.node).tolist()))
                if len(node_wo_dupl) == 0:
                    # no neighbors were sampled
                    break
                duplicates = node.intersection(node_wo_dupl)
                node_wo_dupl.difference_update(duplicates)
                src = Tensor(node_wo_dupl if not self.disjoint else list(
                    zip(*node_wo_dupl))[1]).type(torch.int64)
                node.update(node_wo_dupl)

                node_with_dupl.append(out.node)
                edge.append(out.edge)

                if self.disjoint:
                    src_batch = Tensor(list(zip(*node_wo_dupl))[0]).type(
                        torch.int64)
                    batch_with_dupl.append(out.batch)

                num_sampled_nodes.append(len(src))
                num_sampled_edges.append(len(out.node))
                sampled_nbrs_per_node += out.metadata

            row, col = torch.ops.pyg.relabel_neighborhood(
                seed, torch.cat(node_with_dupl), sampled_nbrs_per_node,
                self._sampler.num_nodes,
                torch.cat(batch_with_dupl) if self.disjoint else None,
                self.csc, self.disjoint)

            node = Tensor(node).type(torch.int64)

            batch, node = node.t().contiguous() if self.disjoint else (None,
                                                                       node)

            sample_output = SamplerOutput(
                node=node, row=row, col=col, edge=torch.cat(edge),
                batch=batch if self.disjoint else None,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges, metadata=metadata)

        return sample_output

    def get_sampled_nbrs_per_node(
        self,
        outputs: List[SamplerOutput],
        seed_size: int,
    ) -> SamplerOutput:
        p_id = self.dist_graph.partition_idx

        outputs[p_id].node = outputs[p_id].node[seed_size:]
        if self.disjoint:
            outputs[p_id].batch = outputs[p_id].batch[seed_size:]

        cumm_sampled_nbrs_per_node = outputs[p_id].metadata

        # do not include seed
        start = np.array(cumm_sampled_nbrs_per_node[1:])
        end = np.array(cumm_sampled_nbrs_per_node[0:-1])

        sampled_nbrs_per_node = list(np.subtract(start, end))

        outputs[p_id].metadata = (sampled_nbrs_per_node)

        return outputs[p_id]

    def merge_sampler_outputs_parallel(
        self,
        partition_ids: Tensor,
        partition_orders: Tensor,
        outputs: List[SamplerOutput],
        one_hop_num: int,
    ) -> SamplerOutput:

        sampled_nodes_with_dupl = [
            o.node if o is not None else None for o in outputs
        ]
        edge_ids = [o.edge if o is not None else None for o in outputs]
        batch = [o.batch if o is not None else None
                 for o in outputs] if self.disjoint else None
        cumm_sampled_nbrs_per_node = [
            o.metadata if o is not None else None for o in outputs
        ]

        partition_ids = partition_ids.tolist()
        partition_orders = partition_orders.tolist()

        partitions_num = self.dist_graph.meta['num_parts']

        out = torch.ops.pyg.merge_sampler_outputs(
            sampled_nodes_with_dupl, edge_ids, batch,
            cumm_sampled_nbrs_per_node, partition_ids, partition_orders,
            partitions_num, one_hop_num, self.disjoint)
        (out_node_with_dupl, out_edge, out_batch,
         out_sampled_nbrs_per_node) = out

        return SamplerOutput(out_node_with_dupl, None, None, out_edge,
                             out_batch if self.disjoint else None,
                             metadata=(out_sampled_nbrs_per_node))

    def merge_sampler_outputs(
        self,
        partition_ids: Tensor,
        outputs: List[SamplerOutput],
    ) -> SamplerOutput:
        r""" Merge neighbor sampler outputs from different partitions
      """
        # TODO: move this function to C++
        partition_ids = partition_ids.tolist()

        node_with_dupl = []
        edge = []
        batch = []
        sampled_nbrs_per_node = []

        p_counters = [0] * self.dist_graph.meta['num_parts']
        cumm_sampled_nbrs_per_node = [
            o.metadata if o is not None else None for o in outputs
        ]

        for p_id in partition_ids:
            if len(cumm_sampled_nbrs_per_node[p_id]) <= 1:
                continue
            begin = cumm_sampled_nbrs_per_node[p_id][p_counters[p_id]]

            begin_edge = begin - cumm_sampled_nbrs_per_node[p_id][0]
            p_counters[p_id] += 1

            end = cumm_sampled_nbrs_per_node[p_id][p_counters[p_id]]
            end_edge = end - cumm_sampled_nbrs_per_node[p_id][0]

            node_with_dupl.append(outputs[p_id].node[begin:end])
            edge.append(outputs[p_id].edge[begin_edge:end_edge])
            batch.append(
                outputs[p_id].batch[begin:end]) if self.disjoint else None

            sampled_nbrs_per_node += [end - begin]

        return SamplerOutput(torch.cat(node_with_dupl), None, None,
                             torch.cat(edge),
                             torch.cat(batch) if self.disjoint else None,
                             metadata=(sampled_nbrs_per_node))

    def merge_sampler_outputs_wo_reorder(
        self,
        outputs: List[SamplerOutput],
    ) -> SamplerOutput:
        r""" Merge neighbor sampler outputs from different partitions
      """
        # TODO: move this function to C++

        sampled_nbrs_per_node = []

        cumm_sampled_nbrs_per_node = [
            o.metadata if o is not None else None for o in outputs
        ]

        node_with_dupl = [o.node for o in outputs]
        edge = [o.edge for o in outputs]
        batch = [o.batch for o in outputs] if self.disjoint else None

        for i in range(len(cumm_sampled_nbrs_per_node)):
            # do not include seed
            start = np.array(cumm_sampled_nbrs_per_node[i][1:])
            end = np.array(cumm_sampled_nbrs_per_node[i][0:-1])

            sampled_nbrs_per_node += list(np.subtract(start, end))

        return SamplerOutput(torch.cat(node_with_dupl), None, None,
                             torch.cat(edge),
                             torch.cat(batch) if self.disjoint else None,
                             metadata=(sampled_nbrs_per_node))

    async def _sample_one_hop(
        self,
        srcs: Tensor,
        one_hop_num: int,
        seed_time: Optional[Tensor] = None,
        batch: OptTensor = None,
        etype: Optional[EdgeType] = None,
    ) -> SamplerOutput:
        r""" Sample one-hop neighbors and induce the coo format subgraph.

      Args:
        srcs: input ids, 1D tensor.
        num_nbr: request(max) number of neighbors for one hop.
        etype: edge type to sample from input ids.

      Returns:
        Tuple[Tensor, Tensor]: unique node ids and edge_index.
      """
        src_ntype = (etype[0] if not self.csc else
                     etype[2]) if etype is not None else None

        partition_ids = self.dist_graph.get_partition_ids_from_nids(
            srcs, src_ntype)
        partition_orders = torch.zeros(len(partition_ids), dtype=torch.long)

        p_outputs: List[SamplerOutput] = [None
                                          ] * self.dist_graph.meta['num_parts']
        futs: List[torch.futures.Future] = []

        local_only = True

        for i in range(self.dist_graph.num_partitions):
            p_id = ((self.dist_graph.partition_idx + i) %
                    self.dist_graph.num_partitions)
            p_mask = (partition_ids == p_id)
            p_srcs = torch.masked_select(srcs, p_mask)
            p_batch = torch.masked_select(
                batch, p_mask) if batch is not None else None
            p_seed_time = torch.masked_select(
                seed_time, p_mask) if seed_time is not None else None

            p_indices = torch.arange(len(p_srcs), dtype=torch.long)
            partition_orders[p_mask] = p_indices

            if p_srcs.shape[0] > 0:
                if p_id == self.dist_graph.partition_idx:

                    p_nbr_out = self._sampler.sample_one_hop(
                        p_srcs, one_hop_num, p_seed_time, p_batch, etype)
                    p_outputs.pop(p_id)
                    p_outputs.insert(p_id, p_nbr_out)
                else:
                    local_only = False
                    to_worker = self.rpc_router.get_to_worker(p_id)
                    futs.append(
                        rpc_async(
                            to_worker, self.rpc_sample_callee_id,
                            args=(p_srcs, one_hop_num, p_seed_time, p_batch,
                                  etype)))

        # Only local nodes were sampled
        if local_only:
            return self.get_sampled_nbrs_per_node(p_outputs, len(srcs))
        # Sampled remote nodes
        res_fut_list = await wrap_torch_future(torch.futures.collect_all(futs))
        for i, res_fut in enumerate(res_fut_list):
            p_outputs.pop(p_id)
            p_outputs.insert(p_id, res_fut.wait())

        return self.merge_sampler_outputs_parallel(partition_ids,
                                                   partition_orders, p_outputs,
                                                   one_hop_num)

    async def _colloate_fn(
        self, output: Union[SamplerOutput, HeteroSamplerOutput]
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        r""" Collect labels and features for the sampled subgrarph if
        necessary, and put them into a sample message.
        """
        if self.is_hetero:
            nlabels = {}
            nfeats = {}
            efeats = {}
            # Collect node labels of input node type.
            node_labels = self.dist_graph.labels
            if node_labels is not None:
                nlabels[self.input_type] = node_labels[output.node[
                    self.input_type]]
            # Collect node features.
            if output.node is not None:
                for ntype in output.node.keys():
                    if output.node[ntype].numel() > 0:
                        fut = self.dist_feature.lookup_features(
                            is_node_feat=True, ids=output.node[ntype],
                            input_type=ntype)
                        print('node fut')
                        print({max(output.node[ntype])},
                              {self.dist_feature.node_feat_pb.size()})
                        nfeat = await wrap_torch_future(fut)
                        nfeat = nfeat.to(torch.device('cpu'))
                        nfeats[ntype] = nfeat
                    else:
                        nfeats[ntype] = None
            # Collect edge features
            if output.edge is not None and self.with_edge_attr:
                for etype in output.edge.keys():
                    if output.edge[etype].numel() > 0:
                        fut = self.dist_feature.lookup_features(
                            is_node_feat=False, ids=output.edge[etype],
                            input_type=etype)
                        print('edge fut')
                        print(
                            f'{max(output.edge[etype])}, {self.dist_feature.edge_feat_pb.size()}'
                        )
                        efeat = await wrap_torch_future(fut)
                        efeat = efeat.to(torch.device('cpu'))
                        efeats[etype] = efeat
                    else:
                        efeats[etype] = None

        else:  # Homo
            # Collect node labels.
            nlabels = self.dist_graph.labels[output.node] if (
                self.dist_graph.labels is not None) else None
            # Collect node features.
            if output.node is not None:
                fut = self.dist_feature.lookup_features(
                    is_node_feat=True, ids=output.node)
                nfeats = await wrap_torch_future(fut)
                nfeats = nfeats.to(torch.device('cpu'))
            # else:
            efeats = None
            # Collect edge features.
            if output.edge is not None and self.with_edge_attr:
                fut = self.dist_feature.lookup_features(
                    is_node_feat=False, ids=output.edge)
                efeats = await wrap_torch_future(fut)
                efeats = efeats.to(torch.device('cpu'))
            else:
                efeats = None

        output.metadata = (*output.metadata, nfeats, nlabels, efeats)
        if self.is_hetero:
            output.row = remap_keys(output.row, self._sampler.to_edge_type)
            output.col = remap_keys(output.col, self._sampler.to_edge_type)
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}()-PID{mp.current_process().pid}"


# Sampling Utilities ##########################################################


def close_sampler(worker_id, sampler):
    # Make sure that mp.Queue is empty at exit and RAM is cleared
    try:
        print(f"Closing event_loop in {repr(sampler)} worker-id {worker_id}")
        sampler.event_loop.shutdown_loop()
    except AttributeError:
        pass
    print(f"Closing rpc in {repr(sampler)} worker-id {worker_id}")
    shutdown_rpc(graceful=True)
