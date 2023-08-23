import torch.multiprocessing as mp
import torch
from torch import Tensor

import copy
from math import ceil
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
from pyparsing import Any
from ordered_set import OrderedSet
import numpy as np
from torch_geometric.sampler.utils import remap_keys

from torch_geometric.sampler import (
  NodeSamplerInput, EdgeSamplerInput,
  NeighborOutput, SamplerOutput, HeteroSamplerOutput,
  NeighborSampler, NegativeSampling, edge_sample_async
)
from torch_geometric.sampler.base import SubgraphType
from ..typing import EdgeType, as_str, NumNeighbors, OptTensor
from ..utils import (
    ensure_device, merge_dict, id2idx,
    merge_hetero_sampler_output, format_hetero_sampler_output,
    id2idx_v2
)

from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore
    )
from torch_geometric.typing import EdgeType, NodeType

from torch_geometric.sampler.base import SubgraphType
from .event_loop import ConcurrentEventLoop, wrap_torch_future
from .rpc import (
  RPCCallBase, rpc_register, rpc_async,
  RPCRouter, 
  rpc_partition_to_workers, shutdown_rpc
)

from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from ..typing import Tuple, Dict

from .dist_context import DistRole, DistContext


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


class RpcSubGraphCallee(RPCCallBase):
  r""" A wrapper for rpc callee that will perform rpc sampling from
  remote processes.
  """
  def __init__(self, sampler: NeighborSampler, device: torch.device):
    super().__init__()
    self.sampler = sampler
    self.device = device

  def rpc_async(self, *args, **kwargs):
    with_edge = kwargs['with_edge']
    output = self.sampler.subgraph_op.node_subgraph(args[0].to(self.device),
                                                    with_edge)
    eids = output.eids.to('cpu') if with_edge else None
    return output.nodes.to('cpu'), output.rows.to('cpu'), output.cols.to('cpu'), eids

  def rpc_sync(self, *args, **kwargs):
      pass

class DistNeighborSampler():
  r""" Asynchronized and distributed neighbor sampler.

  Args:
    data (DistDataset): The graph and feature data with partition info.
    num_neighbors (NumNeighbors): The number of sampling neighbors on each hop.
    with_edge (bool): Whether to sample with edge ids. (default: ``None``).
    collect_features (bool): Whether collect features for sampled results.
      (default: ``None``).
    channel (ChannelBase, optional): The message channel to send sampled
      results. If set to `None`, the sampled results will be returned
      directly with `sample_from_nodes`. (default: ``None``).
    concurrency (int): The max number of concurrent seed batches processed by
      the current sampler. (default: ``1``).
    device: The device to use for sampling. If set to ``None``, the current
      cuda device (got by ``torch.cuda.current_device``) will be used if
      available, otherwise, the cpu device will be used. (default: ``None``).
  """
  def __init__(self,
               current_ctx: DistContext,
               rpc_worker_names: Dict[DistRole, List[str]],
               data: Tuple[LocalGraphStore, LocalFeatureStore],
               #channel: PyGMessageQueue,
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
    assert isinstance(self.dist_graph, LocalGraphStore), f"Provided data is in incorrect format: self.dist_graph must be `LocalGraphStore`, got {type(self.dist_graph)}"
    assert isinstance(self.dist_feature, LocalFeatureStore), f"Provided data is in incorrect format: self.dist_feature must be `LocalFeatureStore`, got {type(self.dist_feature)}"
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
    self.csc = True # always true?


  def register_sampler_rpc(self):  
    
    partition2workers = rpc_partition_to_workers(
    current_ctx = self.current_ctx,
    num_partitions=self.dist_graph.num_partitions,
    current_partition_idx=self.dist_graph.partition_idx
    )
    
    self.rpc_router = RPCRouter(partition2workers)
    self.dist_feature.set_rpc_router(self.rpc_router)
        
    print(f"---- 666.2 -------- register_rpc done    ")

    self._sampler = NeighborSampler(
      data=(self.dist_feature, self.dist_graph),
      num_neighbors=self.num_neighbors,
      subgraph_type=self.subgraph_type,
      replace=self.replace,
      disjoint=self.disjoint,
      temporal_strategy=self.temporal_strategy,
      time_attr=self.time_attr,

    )

    print("----------- DistNeighborSampler: after NeigborSampler()  ------------- ")

    # rpc register
    rpc_sample_callee = RpcSamplingCallee(self._sampler, self.device)
    self.rpc_sample_callee_id = rpc_register(rpc_sample_callee)

    rpc_subgraph_callee = RpcSubGraphCallee(self._sampler, self.device)
    self.rpc_subgraph_callee_id = rpc_register(rpc_subgraph_callee)

    print(f"---- 666.3 -------- register_rpc done    ")
    
  def init_event_loop(self):
    
    print(f"-----{repr(self)}: init_event_loop() Start  ")

    self.event_loop = ConcurrentEventLoop(self.concurrency)
    self.event_loop._loop.call_soon_threadsafe(ensure_device, self.device)
    self.event_loop.start_loop()
    
    print(f"----------- {repr(self)}: init_event_loop()   END ------------- ")

  def sample_from_nodes(
    self,
    inputs: NodeSamplerInput,
    **kwargs
  ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
    r""" Sample multi-hop neighbors from nodes, collect the remote features
    (optional), and send results to the output channel.

    Note that if the output sample channel is specified, this func is
    asynchronized and the sampled result will not be returned directly.
    Otherwise, this func will be blocked to wait for the sampled result and
    return it.

    Args:
      inputs (NodeSamplerInput): The input data with node indices to start
        sampling from.
    """
    #print(f"--------- TTT.2------------inputs={inputs}")
    inputs = NodeSamplerInput.cast(inputs)
    if self.channel is None:
      return self.event_loop.run_task(coro=self._sample_from(self.node_sample,
                                                   inputs))
    cb = kwargs.get('callback', None)
    self.event_loop.add_task(coro=self._sample_from(self.node_sample, inputs),
                  callback=cb)
    #print(f"--------- TTT.3------------")
    return None
    
  def sample_from_edges(
    self,
    inputs: EdgeSamplerInput,
    neg_sampling: Optional[NegativeSampling] = None,
    **kwargs,
  ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
    r""" Sample multi-hop neighbors from edges, collect the remote features
    (optional), and send results to the output channel.

    Note that if the output sample channel is specified, this func is
    asynchronized and the sampled result will not be returned directly.
    Otherwise, this func will be blocked to wait for the sampled result and
    return it.

    Args:
      inputs (EdgeSamplerInput): The input data for sampling from edges
        including the (1) source node indices, the (2) destination node
        indices, the (3) optional edge labels and the (4) input edge type.
    """
    if self.channel is None:
      return self.event_loop.run_task(coro=self._sample_from(edge_sample_async, inputs, self.node_sample,
                                                    self._sampler.num_nodes, self.disjoint,
                                                    self._sampler.node_time,
                                                    neg_sampling, distributed=True))
    cb = kwargs.get('callback', None)
    self.event_loop.add_task(coro=self._sample_from(edge_sample_async, inputs, self.node_sample,
                                                    self._sampler.num_nodes, self.disjoint,
                                                    self._sampler.node_time,
                                                    neg_sampling, distributed=True),
                                                    callback=cb)
    return None

  async def _sample_from(
    self,
    async_func,
    *args, **kwargs
  ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
    
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
    
    # TODO: rm inducer and refactor sampling for hetero

    seed = inputs.node.to(self.device)
    seed_time = inputs.time.to(self.device) if inputs.time is not None else None
    input_type = inputs.input_type
    metadata =  (seed, seed_time, input_type)
    batch_size = seed.numel()
    src_batch = torch.arange(batch_size) if self.disjoint else None

    # print(f" ----777.1 -------- distNSampler:  node_sample, inputs={inputs}, seed={seed}, input_type={input_type} ")
    if self.is_hetero:
      if input_type is None:
        raise ValueError("Input type should be defined")
      
      srcs_dict: Dict[NodeType, Tensor] = {}
      node_dict: Dict[NodeType, OrderedSet[List[int]]] = {}
      node_with_dupl_dict: Dict[NodeType, Tensor] = {}
      edge_dict: Dict[EdgeType, Tensor] = {}
      src_batch_dict: Dict[NodeType, Tensor] = {}
      batch_dict: Dict[NodeType, Tensor] = {}
      batch_with_dupl_dict: Dict[NodeType, Tensor] = {}
      seed_time_dict: Dict[NodeType, Tensor] = {input_type: seed_time}
      sampled_nbrs_per_node_dict: Dict[NodeType, List[int]] = {}
      num_sampled_nodes_dict: Dict[NodeType, List[int]] = {}
      num_sampled_edges_dict: Dict[EdgeType, List[int]] = {}

      for ntype in self._sampler.node_types:
        srcs_dict.update({ntype: torch.empty(0, dtype=torch.int64)})
        node_dict.update({ntype: OrderedSet([])})
        node_with_dupl_dict.update({ntype: torch.empty(0, dtype=torch.int64)})
        batch_dict.update({ntype: torch.empty(0, dtype=torch.int64)}) if self.disjoint else None
        src_batch_dict.update({ntype: torch.empty(0, dtype=torch.int64)}) if self.disjoint else None
        batch_with_dupl_dict.update({ntype: torch.empty(0, dtype=torch.int64)}) if self.disjoint else None
        sampled_nbrs_per_node_dict.update({ntype: []})
        num_sampled_nodes_dict.update({ntype: [0]})

      srcs_dict[input_type] = seed
      src_batch_dict[input_type] = src_batch

      edge_types = []
      node_types = []
      for etype in self._sampler.edge_types:
        edge_dict.update({etype: torch.empty(0, dtype=torch.int64)})
        num_sampled_edges_dict.update({etype: []})

        src = etype[0] if not self.csc else etype[2]
        dst = etype[2] if not self.csc else etype[0]
        if srcs_dict.get(src, None).numel():
          edge_types.append(etype)
          node_types.extend([src, dst])
        node_types = list(set(node_types))
          
      node_dict[input_type] = OrderedSet(seed.tolist()) if not self.disjoint else OrderedSet(tuple(zip(src_batch.tolist(), seed.tolist())))
      num_sampled_nodes_dict[input_type].append(seed.numel())

      for i in range(self._sampler.num_hops):
        task_dict = {}
        for etype in edge_types:
          src = etype[0] if not self.csc else etype[2]
          srcs = srcs_dict.get(src, None)
          seed_time = seed_time_dict.get(src, None) if seed_time_dict.get(src, None) is not None else None
          one_hop_num = self.num_neighbors[i] if isinstance(self.num_neighbors, List) else self.num_neighbors[etype][i]
          task_dict[etype] = self.event_loop._loop.create_task(
            self._sample_one_hop(srcs, one_hop_num, seed_time, src_batch_dict[src], etype))
        for etype, task in task_dict.items():
          out: HeteroSamplerOutput = await task

          # remove duplicates
          # TODO: find better method to remove duplicates
          node_wo_dupl = OrderedSet((out.node).tolist()) if not self.disjoint else OrderedSet(zip((out.batch).tolist(), (out.node).tolist()))
          if len(node_wo_dupl) == 0:
          # no neighbors were sampled
            break
          dst = etype[2] if not self.csc else etype[0]
          duplicates = node_dict[dst].intersection(node_wo_dupl)
          node_wo_dupl.difference_update(duplicates)
          srcs_dict[dst] = torch.Tensor(node_wo_dupl if not self.disjoint else list(zip(*node_wo_dupl))[1]).type(torch.int64)
          node_dict[dst].update(node_wo_dupl)

          node_with_dupl_dict[dst] = torch.cat([node_with_dupl_dict[dst], out.node])
          edge_dict[etype] = torch.cat([edge_dict[etype], out.edge])

          if self.disjoint:
            src_batch_dict[dst] = torch.Tensor(list(zip(*node_wo_dupl))[0]).type(torch.int64)
            batch_with_dupl_dict[dst] = torch.cat([batch_with_dupl_dict[dst], out.batch])

          num_sampled_nodes_dict[dst].append(len(srcs_dict[dst]))
          num_sampled_edges_dict[etype].append(len(out.node))
          sampled_nbrs_per_node_dict[dst] += out.metadata

      row_dict, col_dict = torch.ops.pyg.hetero_relabel_neighborhood(node_types, edge_types, {input_type: seed}, node_with_dupl_dict, sampled_nbrs_per_node_dict, self._sampler.num_nodes, batch_with_dupl_dict, self.csc, self.disjoint)

      
      node_dict = {ntype: torch.Tensor(node_dict[ntype]).type(torch.int64) for ntype in self._sampler.node_types}
      if self.disjoint:
          for ntype in node_types:
            batch_dict[ntype], node_dict[ntype] = node_dict[ntype].t().contiguous()

      sample_output = HeteroSamplerOutput(
        node=node_dict,
        row=row_dict,
        col=col_dict,
        edge=edge_dict,
        batch=batch_dict if self.disjoint else None,
        num_sampled_nodes=num_sampled_nodes_dict,
        num_sampled_edges=num_sampled_edges_dict,
        metadata=metadata
      )
    else:

      srcs = seed

      node = OrderedSet(srcs.tolist()) if not self.disjoint else OrderedSet(tuple(zip(src_batch.tolist(), srcs.tolist())))
      node_with_dupl = torch.empty(0, dtype=torch.int64)
      batch = torch.empty(0, dtype=torch.int64) if self.disjoint else None
      batch_with_dupl = torch.empty(0, dtype=torch.int64)
      edge = torch.empty(0, dtype=torch.int64)
      
      sampled_nbrs_per_node = []
      num_sampled_nodes = [seed.numel()]
      num_sampled_edges = [0]

      for one_hop_num in self.num_neighbors:
        out = await self._sample_one_hop(srcs, one_hop_num, seed_time, src_batch)

        # remove duplicates
        # TODO: find better method to remove duplicates
        node_wo_dupl = OrderedSet((out.node).tolist()) if not self.disjoint else OrderedSet(zip((out.batch).tolist(), (out.node).tolist()))
        if len(node_wo_dupl) == 0:
          # no neighbors were sampled
          break
        duplicates = node.intersection(node_wo_dupl)
        node_wo_dupl.difference_update(duplicates)
        srcs = torch.Tensor(node_wo_dupl if not self.disjoint else list(zip(*node_wo_dupl))[1]).type(torch.int64)
        node.update(node_wo_dupl)

        node_with_dupl = torch.cat([node_with_dupl, out.node])

        edge = torch.cat([edge, out.edge])

        if self.disjoint:
          src_batch = torch.Tensor(list(zip(*node_wo_dupl))[0]).type(torch.int64)
          batch_with_dupl = torch.cat([batch_with_dupl, out.batch])

        num_sampled_nodes.append(len(srcs))
        num_sampled_edges.append(len(out.node))
        sampled_nbrs_per_node += out.metadata

      row, col = torch.ops.pyg.relabel_neighborhood(seed, node_with_dupl, sampled_nbrs_per_node, self._sampler.num_nodes, batch_with_dupl, self.csc, self.disjoint)
      # print("sampled nbrs per node: ")
      # print(sampled_nbrs_per_node)
      # print("row:")
      # print(row)
      # print("col:")
      # print(col)

      node = torch.Tensor(node).type(torch.int64)
      if self.disjoint:
        batch, node = node.t().contiguous()

      sample_output = SamplerOutput(
        node=node,
        row=row,
        col=col,
        edge=edge,
        batch=batch if self.disjoint else None,
        num_sampled_nodes=num_sampled_nodes,
        num_sampled_edges=num_sampled_edges,
        metadata=metadata
       )

    return sample_output

  def form_local_output(
    self,
    results: List[SamplerOutput],
  ) -> SamplerOutput:
    p_id = self.dist_graph.partition_idx

    cumm_sampled_nbrs_per_node = results[p_id].metadata

    # do not include seed
    start = np.array(cumm_sampled_nbrs_per_node[1:])
    end = np.array(cumm_sampled_nbrs_per_node[0:-1])

    sampled_nbrs_per_node = list(np.subtract(start, end))

    results[p_id].metadata = (sampled_nbrs_per_node)

    return results[p_id]
    

  def merge_sampler_outputs(
    self,
    partition_ids: torch.Tensor,
    outputs: List[SamplerOutput],
  ) -> SamplerOutput:
    r""" Merge neighbor sampler outputs from different partitions
    """
    # TODO: move this function to C++

    partition_ids = partition_ids.tolist()

    node_with_dupl = torch.empty(0, dtype=torch.int64)
    edge = torch.empty(0, dtype=torch.int64)
    batch = torch.empty(0, dtype=torch.int64) if self.disjoint else None
    sampled_nbrs_per_node = []

    p_counters = [0] * self.dist_graph.meta['num_parts']
    cumm_sampled_nbrs_per_node = [o.metadata if o is not None else None for o in outputs]

    for p_id in partition_ids:
      if len(cumm_sampled_nbrs_per_node[p_id]) <= 1:
        continue
      start = cumm_sampled_nbrs_per_node[p_id][p_counters[p_id]]
      p_counters[p_id] += 1
      end = cumm_sampled_nbrs_per_node[p_id][p_counters[p_id]]

      node_with_dupl = torch.cat([node_with_dupl, outputs[p_id].node[start: end]])
      edge = torch.cat([edge, outputs[p_id].edge[start: end]])
      batch = torch.cat([batch, outputs[p_id].batch[start: end]]) if self.disjoint else None

      sampled_nbrs_per_node += [end - start]
    
    #print(f"--------YYY.5   --- sampled_nbrs_per_node={sampled_nbrs_per_node}, node_with_dupl={node_with_dupl} ")
    #print(f"------77777.3------  merge_results --------- ")
    return SamplerOutput(
      node_with_dupl,
      None,
      None,
      edge,
      batch,
      metadata=(sampled_nbrs_per_node)
    )
      

  async def _sample_one_hop(
    self,
    srcs: Tensor,
    one_hop_num: int,
    seed_time: Optional[Tensor] = None,
    batch: OptTensor = None,
    edge_type: Optional[EdgeType] = None,
  ) -> SamplerOutput:
    r""" Sample one-hop neighbors and induce the coo format subgraph.

    Args:
      srcs: input ids, 1D tensor.
      num_nbr: request(max) number of neighbors for one hop.
      etype: edge type to sample from input ids.

    Returns:
      Tuple[torch.Tensor, torch.Tensor]: unique node ids and edge_index.
    """
    src_ntype = (edge_type[0] if not self.csc else edge_type[2]) if edge_type is not None else None
    
    partition_ids = self.dist_graph.get_partition_ids_from_nids(srcs, src_ntype)

    p_outputs: List[SamplerOutput] = [None] * self.dist_graph.meta['num_parts']
    futs: List[torch.futures.Future] = []

    local_only = True

    for i in range(self.dist_graph.num_partitions):
      p_id = (
        (self.dist_graph.partition_idx + i) % self.dist_graph.num_partitions
      )
      p_mask = (partition_ids == p_id)
      p_srcs = torch.masked_select(srcs, p_mask)
      p_batch = torch.masked_select(batch, p_mask) if batch is not None else None
      p_seed_time = torch.masked_select(seed_time, p_mask) if seed_time is not None else None

      if p_srcs.shape[0] > 0:
        if p_id == self.dist_graph.partition_idx:
          
          p_nbr_out = self._sampler.sample_one_hop(p_srcs, one_hop_num, p_seed_time, p_batch, edge_type)
          p_outputs.pop(p_id)
          p_outputs.insert(p_id, p_nbr_out)
        else:
          local_only = False
          to_worker = self.rpc_router.get_to_worker(p_id)
          futs.append(rpc_async(to_worker,
                                self.rpc_sample_callee_id,
                                args=(p_srcs, one_hop_num, p_seed_time, p_batch, edge_type)))

    # Only local nodes were sampled
    if local_only:
      return self.form_local_output(p_outputs)
    # Sampled remote nodes
    res_fut_list = await wrap_torch_future(torch.futures.collect_all(futs))
    for i, res_fut in enumerate(res_fut_list):
      p_outputs.pop(p_id)
      p_outputs.insert(p_id, res_fut.wait())

    return self.merge_sampler_outputs(partition_ids, p_outputs)

  async def _colloate_fn(
    self,
    output: Union[SamplerOutput, HeteroSamplerOutput]
  ) -> Union[SamplerOutput, HeteroSamplerOutput]:
    r""" Collect labels and features for the sampled subgrarph if necessary,
    and put them into a sample message.
    """
    result_map = {}

    input_type = output.metadata[2]
    
    if self.is_hetero:
      nlabels = {}
      nfeats = {}
      efeats = {}
      
      # Collect node labels of input node type.
      if not isinstance(input_type, Tuple):
        node_labels = self.dist_graph.labels
        if node_labels is not None:
          nlabels[f'{as_str(input_type)}'] = \
            node_labels[output.node[input_type]]
      # Collect node features.
      
      for ntype in output.node.keys():
        fut = self.dist_feature.lookup_features(is_node_feat=True, ids=output.node[ntype], input_type=ntype)
        nfeat = await wrap_torch_future(fut)
        nfeat = nfeat.to(torch.device('cpu'))
        nfeats[ntype] = nfeat
        
      # Collect edge features
      for etype in output.edge.keys():
        try:
          fut = self.dist_feature.lookup_features(is_node_feat=False, ids=output.edge[etype], input_type=etype)
          efeat = await wrap_torch_future(fut)
          efeat = efeat.to(torch.device('cpu'))
          efeats[etype] = efeat
        except KeyError:
          efeats[etype] = None
        
    else:
        # Collect node labels.
        nlabels = self.dist_graph.labels[output.node] if (self.dist_graph.labels is not None) else None
        # Collect node features.
        if output.node is not None:
          fut = self.dist_feature.lookup_features(is_node_feat=True, ids=output.node)
          nfeats = await wrap_torch_future(fut) 
          nfeats = nfeats.to(torch.device('cpu'))
        # else:
        efeats = None
        # Collect edge features.
        # try:
        #   # fut = self.dist_feature.lookup_features(is_node_feat=False, ids=output.edge)
        #   # efeats = await wrap_torch_future(fut)
        #   # efeats = efeats.to(torch.device('cpu'))
        # except KeyError: 
        #   efeats = None

    #print(f"------- 777.4 ----- DistNSampler: _colloate_fn()  return -------")
    output.metadata = (output.metadata[0], output.metadata[1], nfeats, nlabels, efeats)
    return output #result_map

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
  
