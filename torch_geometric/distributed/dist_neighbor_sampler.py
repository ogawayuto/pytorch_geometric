import torch.multiprocessing as mp
import torch
from torch import Tensor

import copy
from math import ceil
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
from pyparsing import Any
from ordered_set import OrderedSet


##from .. import py_graphlearn_torch as pywrap
#from ..channel import ChannelBase, SampleMessage
from torch_geometric.sampler import (
  NodeSamplerInput, EdgeSamplerInput,
  NeighborOutput, SamplerOutput, HeteroSamplerOutput,
  NeighborSampler, NegativeSampling, edge_sample_async
)
from torch_geometric.sampler.base import SubgraphType
from ..typing import EdgeType, as_str, NumNeighbors, OptTensor
from ..utils import (
    #get_available_device, 
    ensure_device, merge_dict, id2idx,
    merge_hetero_sampler_output, format_hetero_sampler_output,
    id2idx_v2
)

from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore
    )
from torch_geometric.typing import EdgeType, NodeType

#from .dist_dataset import DistDataset
#from .dist_feature import DistFeature
#from .dist_graph import DistGraph

from torch_geometric.sampler.base import SubgraphType
from .event_loop import ConcurrentEventLoop, wrap_torch_future
from .rpc import (
  RPCCallBase, rpc_register, rpc_async,
  RPCRouter, 
  rpc_partition_to_workers, shutdown_rpc
)
##
from torch_geometric.data import Data
from torch_geometric.data import TensorAttr

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
    #ensure_device(self.device)
    output = self.sampler.sample_one_hop(*args, **kwargs)

    #if(output.device=='cpu'):
    return output
    #else:
    #    return output.to(torch.device('cpu'))


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
    #ensure_device(self.device)
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
               channel: mp.Queue(),
               num_neighbors: Optional[NumNeighbors] = None,
               replace: bool = False,
               subgraph_type: Union[SubgraphType, str] = 'directional',
               disjoint: bool = True,
               temporal_strategy: str = 'uniform',
               time_attr: Optional[str] = None,
               with_edge: bool = False,
               collect_features: bool = False,
               concurrency: int = 1,
               device: Optional[torch.device] = None,
               **kwargs,
               ):
    print(f"---- 555.1 -------- data={data}     ")
    self.current_ctx = current_ctx
    self.rpc_worker_names = rpc_worker_names

    self.data = data
    self.graph = data[0]
    self.feature = data[1]

    self.num_neighbors = num_neighbors
    self.with_edge = with_edge
    self.collect_features = collect_features
    self.channel = channel
    self.concurrency = concurrency
    self.device = device
    self.event_loop = None
    self.replace = replace
    self.subgraph_type = subgraph_type
    self.disjoint = disjoint
    self.temporal_strategy = temporal_strategy
    self.time_attr = time_attr


  def register_sampler_rpc(self):  
    
    if True:
      partition2workers = rpc_partition_to_workers(
        current_ctx = self.current_ctx,
        num_partitions=self.graph.num_partitions, #self.data[1],
        current_partition_idx=self.graph.partition_idx #self.data[2], #.partition_idx
      )
      
      self.rpc_router = RPCRouter(partition2workers)
      self.dist_graph = self.graph


      print(f"----////////88888//////////---- self.dist_graph.num_partitions={self.dist_graph.num_partitions}, self.dist_graph.node_pb={self.dist_graph.node_pb}, self.dist_graph.meta={self.dist_graph.meta}    ")
      
      # edge_index = self.graph.get_edge_index(edge_type=None, layout='coo')
      # print(f"----////////88888//////////---- edge_index={edge_index}    ")

      self.dist_node_feature = None
      self.dist_edge_feature = None
      if self.collect_features:
        if self.dist_graph.meta["is_hetero"]:
          attrs = self.feature.get_all_tensor_attrs()
          x_attrs = [
                    copy.copy(attr) for attr in attrs
                    if attr.attr_name == 'x'
          ]
          node_features = self.feature.multi_get_tensor(x_attrs)
        else:
          node_features = self.feature.get_tensor(group_name=None, attr_name='x')
          print(f"--000000000000.1-- node_features={node_features} ")
        
      
        if node_features is not None:
            local_feature=self.feature
            local_feature.set_rpc_router(self.rpc_router)
            
            self.dist_node_feature = local_feature

        edge_features = None

        if self.with_edge and edge_features is not None:
          self.dist_edge_feature = None

    else:
      raise ValueError(f"'{self.__class__.__name__}': found invalid input "
                       f"data type '{type(data)}'")

    print(f"---- 666.2 -------- register_rpc done    ")

    self._sampler = NeighborSampler(
      data=(self.dist_node_feature, self.dist_graph),
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
  ) -> Optional[Dict[str, torch.Tensor]]:
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
    print(f"--------- TTT.2------------inputs={inputs}")
    inputs = NodeSamplerInput.cast(inputs)
    if self.channel is None:
      return self.event_loop.run_task(coro=self._send_adapter(self.node_sample,
                                                   inputs))
    cb = kwargs.get('callback', None)
    self.event_loop.add_task(coro=self._send_adapter(self.node_sample, inputs),
                  callback=cb)
    print(f"--------- TTT.3------------")
    return None
    
  def sample_from_edges(
    self,
    inputs: EdgeSamplerInput,
    neg_sampling: Optional[NegativeSampling] = None,
    **kwargs,
  ) -> Optional[Dict[str, torch.Tensor]]:
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
      return self.event_loop.run_task(coro=self._send_adapter(edge_sample_async, inputs, self.node_sample,
                                                    self._sampler.num_nodes, self.disjoint,
                                                    self._sampler.node_time,
                                                    neg_sampling, distributed=True))
    cb = kwargs.get('callback', None)
    self.event_loop.add_task(coro=self._send_adapter(edge_sample_async, inputs, self.node_sample,
                                                    self._sampler.num_nodes, self.disjoint,
                                                    self._sampler.node_time,
                                                    neg_sampling, distributed=True),
                                                    callback=cb)
    return None

  def subgraph(
    self,
    inputs: NodeSamplerInput,
    **kwargs
  ) -> Optional[Dict[str, torch.Tensor]]:
    r""" Induce an enclosing subgraph based on inputs and their neighbors(if
      self.num_neighbors is not None).
    """
    inputs = NodeSamplerInput.cast(inputs)
    if self.channel is None:
      return self.run_task(coro=self._send_adapter(self._subgraph, inputs))
    cb = kwargs.get('callback', None)
    self.add_task(coro=self._send_adapter(self._subgraph, inputs), callback=cb)
    return None

  async def _send_adapter(
    self,
    async_func,
    *args, **kwargs
  ) -> Union[SamplerOutput, HeteroSamplerOutput]:
    
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

    print(f" ----777.1 -------- distNSampler:  node_sample, inputs={inputs}, seed={seed}, input_type={input_type} ")
    if(self.dist_graph.meta["is_hetero"]):
      if input_type is None:
        raise ValueError("Input type should be defined")
      srcs_dict = {input_type: seed}
      batch_size = srcs_dict[input_type].numel()

      node = {input_type: OrderedSet(seed)}
      node_with_dupl = {input_type: torch.empty(0, dtype=torch.int64)}
      edge = {input_type: torch.empty(0, dtype=torch.int64)} # should be edge_type?

      sampled_nbrs_per_node_dict = {}
      num_sampled_nodes_dict = {input_type: seed.numel()}
      num_sampled_edges_dict = {input_type: 0} # should be edge type?

      for i in range(self._sampler.num_hops):
        task_dict, nbr_dict, edge_dict = {}, {}, {} # ??
        for etype in self._sampler.edge_types:
          srcs = srcs_dict.get(etype[2], None)
          one_hop_num = {etype: [self.num_neighbors[i]]} if isinstance(self.num_neighbors, List) else self.num_neighbors[etype][i]
          if srcs is not None:
            task_dict[etype] = self.event_loop._loop.create_task(
              self._sample_one_hop(srcs, one_hop_num, None, None, etype)) # add seed_time and src_batch
        for etype, task in task_dict.items():
          out: HeteroSamplerOutput = await task

        nodes_dict, rows_dict, cols_dict = inducer.induce_next(nbr_dict)
        merge_dict(nodes_dict, out_nodes)
        merge_dict(rows_dict, out_rows)
        merge_dict(cols_dict, out_cols)
        merge_dict(edge_dict, out_edges)
        src_dict = nodes_dict

      sample_output = HeteroSamplerOutput(
        node={ntype: torch.cat(nodes) for ntype, nodes in out_nodes.items()},
        row={etype: torch.cat(rows) for etype, rows in out_rows.items()},
        col={etype: torch.cat(cols) for etype, cols in out_cols.items()},
        edge=(
          {etype: torch.cat(eids) for etype, eids in out_edges.items()}
          if self.with_edge else None
        ),
        metadata={'input_type': input_type, 'batch_size': batch_size}
      )
    else:

      srcs = seed
      batch_size = seed.numel()
      src_batch = torch.arange(batch_size) if self.disjoint else None
      batch = src_batch

      node = OrderedSet(srcs) if not self.disjoint else OrderedSet(tuple(zip(batch, srcs)))
      node_with_dupl = torch.empty(0, dtype=torch.int64)
      edge = torch.empty(0, dtype=torch.int64)
      
      sampled_nbrs_per_node = []
      num_sampled_nodes = [seed.numel()]
      num_sampled_edges = [0]

      for one_hop_num in self.num_neighbors:
        out = await self._sample_one_hop(srcs, one_hop_num, seed_time, src_batch)

        # remove duplicates
        # TODO: find better method to remove duplicates
        node_wo_dupl = OrderedSet(out.node) if not self.disjoint else OrderedSet(zip(out.batch, out.node))
        if len(node_wo_dupl) == 0:
          # no neighbors were sampled
          break
        duplicates = node.intersection(node_wo_dupl)
        node_wo_dupl.difference_update(duplicates)
        srcs = torch.Tensor(node_wo_dupl if not self.disjoint else list(zip(*node_wo_dupl))[1]).type(torch.int64)
        node.update(node_wo_dupl)

        node_with_dupl = torch.cat([node_with_dupl, out.node])
        edge = torch.cat([edge, out.edge]) if self.with_edge else None
        src_batch = out.batch
        batch = torch.cat([batch, out.batch]) if self.disjoint else None
        num_sampled_nodes.append(len(srcs))
        num_sampled_edges.append(len(out.node))
        sampled_nbrs_per_node += out.metadata

      row, col = torch.ops.pyg.get_adj_matrix(seed, node_with_dupl, sampled_nbrs_per_node, self._sampler.num_nodes, self.disjoint)
      # print("sampled nbrs per node: ")
      # print(sampled_nbrs_per_node)
      # print("row:")
      # print(row)
      # print("col:")
      # print(col)

      node = torch.Tensor(node).type(torch.int64)
      if self.disjoint:
        node = node.t().contiguous()

      sample_output = SamplerOutput(
        node=node,
        row=row,
        col=col,
        edge=edge if self.with_edge else None,
        batch=batch if self.disjoint else None,
        num_sampled_nodes=num_sampled_nodes if num_sampled_nodes != None else None,
        num_sampled_edges=num_sampled_edges if num_sampled_edges != None else None,
        metadata={'input_type': None, 'batch_size': batch_size}
       )

    return sample_output


  async def edge_sample_old_to_be_removed(
      self,
      inputs: EdgeSamplerInput,
      neg_sampling: Optional[NegativeSampling] = None,
  ) -> Optional[Dict[str, torch.Tensor]]:
    r"""Performs sampling from an edge sampler input, leveraging a sampling
    function of the same signature as `node_sample`.

    Currently, we support the out-edge sampling manner, so we reverse the
    direction of src and dst for the output so that features of the sampled
    nodes during training can be aggregated from k-hop to (k-1)-hop nodes.

    Note: Negative sampling is performed locally and unable to fetch positive
    edges from remote, so the negative sampling in the distributed case is
    currently non-strict for both binary and triplet manner.
    """
    # TODO: refactor
    src = inputs.row.to(self.device)
    dst = inputs.col.to(self.device)
    edge_label = None if inputs.label is None else inputs.label.to(self.device)
    input_type = inputs.input_type

    assert isinstance(self._sampler.num_nodes, (dict, int))
    if not isinstance(self._sampler.num_nodes, dict):
        num_src_nodes = num_dst_nodes = self._sampler.num_nodes
    else:
        num_src_nodes = self._sampler.num_nodes[input_type[0]]
        num_dst_nodes = self._sampler.num_nodes[input_type[-1]]

    num_pos = src.numel()
    num_neg = 0
    # Negative Sampling
    self._sampler.lazy_init_neg_sampler() # rm?
    if neg_sampling is not None:
      # When we are doing negative sampling, we append negative information
      # of nodes/edges to `src`, `dst`.
      # Later on, we can easily reconstruct what belongs to positive and
      # negative examples by slicing via `num_pos`.
      num_neg = ceil(num_pos * neg_sampling.amount)
      if neg_sampling.is_binary():
        # In the "binary" case, we randomly sample negative pairs of nodes.
        if input_type is not None:
          neg_pair = self._sampler._neg_sampler[input_type].sample(num_neg)
        else:
          src_neg = neg_sample(src, neg_sampling, num_src_nodes, None, None).to(self.device) # TODO add time args
          src = torch.cat([src, src_neg], dim=0)

          dst_neg = neg_sample(dst, neg_sampling, num_dst_nodes, None, None).to(self.device) # TODO add time args
          dst = torch.cat([dst, dst_neg], dim=0)

        if edge_label is None:
            edge_label = torch.ones(num_pos, device=self.device)
        size = (num_neg, ) + edge_label.size()[1:]
        edge_neg_label = edge_label.new_zeros(size)
        edge_label = torch.cat([edge_label, edge_neg_label])

      elif neg_sampling.is_triplet():
        # In the "triplet" case, we randomly sample negative destinations.
        assert num_neg % num_pos == 0 # ??
        if input_type is not None:
          neg_pair = self._sampler._neg_sampler[input_type].sample(num_neg, padding=True)
        else:
          dst_neg = neg_sample(dst, neg_sampling, num_dst_nodes, None, None).to(self.device) # TODO add time args
          dst = torch.cat([dst, dst_neg], dim=0)

        assert edge_label is None

    # Neighbor Sampling
    if input_type is not None: # hetero
      if input_type[0] != input_type[-1]:  # Two distinct node types:
        src_seed, dst_seed = src, dst
        src, inverse_src = src.unique(return_inverse=True)
        dst, inverse_dst = dst.unique(return_inverse=True)
        seed_dict = {input_type[0]: src, input_type[-1]: dst}
      else:  # Only a single node type: Merge both source and destination.
        seed = torch.cat([src, dst], dim=0)
        seed, inverse_seed = seed.unique(return_inverse=True)
        seed_dict = {input_type[0]: seed}

      temp_out = []
      for it, node in seed_dict.items():
        seeds = NodeSamplerInput(node=node, input_type=it)
        temp_out.append(await self.node_sample(seeds))
      if len(temp_out) == 2:
        out = merge_hetero_sampler_output(temp_out[0],
                                          temp_out[1],
                                          device=self.device)
      else:
        out = format_hetero_sampler_output(temp_out[0])

      # edge_label
      if neg_sampling is None or neg_sampling.is_binary():
        if input_type[0] != input_type[-1]:
          inverse_src = id2idx_v2(src_seed, out.node[input_type[0]])
          inverse_dst = id2idx_v2(dst_seed, out.node[input_type[-1]])
          edge_label_index = torch.stack([
              inverse_src,
              inverse_dst,
          ], dim=0)
        else:
          edge_label_index = inverse_seed.view(2, -1)

        out.metadata.update({'edge_label_index': edge_label_index,
                             'edge_label': edge_label})
        out.input_type = input_type
      elif neg_sampling.is_triplet():
        if input_type[0] != input_type[-1]:
          inverse_src = id2idx_v2(src_seed, out.node[input_type[0]])
          inverse_dst = id2idx_v2(dst_seed, out.node[input_type[-1]])
          src_index = inverse_src
          dst_pos_index = inverse_dst[:num_pos]
          dst_neg_index = inverse_dst[num_pos:]
        else:
          src_index = inverse_seed[:num_pos]
          dst_pos_index = inverse_seed[num_pos:2 * num_pos]
          dst_neg_index = inverse_seed[2 * num_pos:]
        dst_neg_index = dst_neg_index.view(num_pos, -1).squeeze(-1)

        out.metadata.update({'src_index': src_index,
                             'dst_pos_index': dst_pos_index,
                             'dst_neg_index': dst_neg_index})
        out.input_type = input_type
    else: #homo
      seed = torch.cat([src, dst], dim=0)
      seed, inverse_seed = seed.unique(return_inverse=True)
      out = await self.node_sample(NodeSamplerInput(None, seed, None, None))

      # edge_label
      if neg_sampling is None or neg_sampling.is_binary():
        edge_label_index = inverse_seed.view(2, -1)

        out.metadata.update({'edge_label_index': edge_label_index,
                             'edge_label': edge_label})
      elif neg_sampling.is_triplet():
        src_index = inverse_seed[:num_pos]
        dst_pos_index = inverse_seed[num_pos:2 * num_pos]
        dst_neg_index = inverse_seed[2 * num_pos:]
        dst_neg_index = dst_neg_index.view(num_pos, -1).squeeze(-1)

        # out.metadata = (input_id, src_index, dst_pos_index, dst_neg_index,
        #                     src_time)
        out.metadata.update({'src_index': src_index,
                             'dst_pos_index': dst_pos_index,
                             'dst_neg_index': dst_neg_index})

    return out

  async def _subgraph(
    self,
    inputs: NodeSamplerInput,
  ) -> Optional[Dict[str, torch.Tensor]]:
    #TODO: refactor
    inputs = NodeSamplerInput.cast(inputs)
    input_seeds = inputs.node.to(self.device)
    #is_hetero = (self.dist_graph.data_cls == 'hetero')
    #if is_hetero:
    if self.dist_graph.meta["is_hetero"]:
      raise NotImplementedError
    else:
      # neighbor sampling.
      if self.num_neighbors is not None:
        nodes = [input_seeds]
        for num in self.num_neighbors:
          nbr = await self._sample_one_hop(nodes[-1], num)
          nodes.append(torch.unique(nbr.nbr))
        nodes = torch.cat(nodes)
      else:
        nodes = input_seeds
      nodes, mapping = torch.unique(nodes, return_inverse=True)
      nid2idx = id2idx(nodes)
      # subgraph inducing.
      partition_ids = self.dist_graph.get_node_partitions(nodes)
      partition_ids = partition_ids.to(self.device)
      rows, cols, eids, futs = [], [], [], []
      for i in range(self.data.num_partitions):
        p_id = (self.data.partition_idx + i) % self.data.num_partitions
        p_ids = torch.masked_select(nodes, (partition_ids == p_id))
        if p_ids.shape[0] > 0:
          if p_id == self.data.partition_idx:
            subgraph = self._sampler.subgraph_op.node_subgraph(nodes, self.with_edge)
            # relabel row and col indices.
            rows.append(nid2idx[subgraph.nodes[subgraph.rows]])
            cols.append(nid2idx[subgraph.nodes[subgraph.cols]])
            if self.with_edge:
              eids.append(subgraph.eids.to(self.device))
          else:
            to_worker = self.rpc_router.get_to_worker(p_id)
            futs.append(rpc_async(to_worker,
                                          self.rpc_subgraph_callee_id,
                                          args=(nodes.cpu(),),
                                          kwargs={'with_edge': self.with_edge}))
      if not len(futs) == 0:
        res_fut_list = await wrap_torch_future(torch.futures.collect_all(futs))
        for res_fut in res_fut_list:
          res_nodes, res_rows, res_cols, res_eids = res_fut.wait()
          res_nodes = res_nodes.to(self.device)
          rows.append(nid2idx[res_nodes[res_rows]])
          cols.append(nid2idx[res_nodes[res_cols]])
          if self.with_edge:
            eids.append(res_eids.to(self.device))

      sample_output = SamplerOutput(
        node=nodes,
        row=torch.cat(rows),
        col=torch.cat(cols),
        edge=torch.cat(eids) if self.with_edge else None,
        device=self.device,
        metadata={'mapping': mapping[:input_seeds.numel()]})

      return sample_output

  def merge_results(
    self,
    partition_ids: torch.Tensor,
    results: List[Union[SamplerOutput, HeteroSamplerOutput]],
    edge_type: EdgeType = None
  ) -> Union[SamplerOutput, HeteroSamplerOutput]:
    r""" Merge neighbor sampler outputs from different devices into a complete one.
    """
    partition_ids = partition_ids.tolist()
    

    node_with_dupl = torch.empty(0, dtype=torch.int64)
    edge = torch.empty(0, dtype=torch.int64) if self.with_edge else None
    batch = torch.empty(0, dtype=torch.int64) if self.disjoint else None
    sampled_nbrs_per_node = []

    p_counters = [0] * self.dist_graph.meta['num_parts'] if self.dist_graph.meta['is_hetero'] and edge_type[0] != edge_type[2] else [1] * self.dist_graph.meta['num_parts']

    if self.dist_graph.meta['is_hetero']:
      cumm_sampled_nbrs_per_node = [r.metadata if r is not None else None for r in results]
        
      dst = edge_type[0] # if csc

      for p_id in partition_ids:
        if edge_type[0] == edge_type[1]:
          if len(cumm_sampled_nbrs_per_node[p_id][dst]) <= 2:
            continue
        else:
          if len(cumm_sampled_nbrs_per_node[p_id][dst]) <= 1:
            continue

        start = cumm_sampled_nbrs_per_node[p_id][dst][p_counters[p_id]]
        p_counters[p_id] += 1
        end = cumm_sampled_nbrs_per_node[p_id][dst][p_counters[p_id]]

        node_with_dupl = torch.cat([node_with_dupl, results[p_id].node[dst][start: end]])
        edge = torch.cat([edge, results[p_id].edge[edge_type][start: end]]) if self.with_edge else None
        batch = torch.cat([batch, results[p_id].batch[dst][start: end]]) if self.disjoint else None

        sampled_nbrs_per_node += [end - start]
      
      return HeteroSamplerOutput(
        node={dst: node_with_dupl},
        row=None,
        col=None,
        edge={edge_type: edge},
        batch={dst: batch},
        metadata=({dst: sampled_nbrs_per_node})
      )

    else:
      # do not include seed
      cumm_sampled_nbrs_per_node = [r.metadata if r is not None else None for r in results]

      node_with_dupl = torch.empty(0, dtype=torch.int64)
      edge = torch.empty(0, dtype=torch.int64) if self.with_edge else None
      batch = torch.empty(0, dtype=torch.int64) if self.disjoint else None
      sampled_nbrs_per_node = []

      for p_id in partition_ids:
          if len(cumm_sampled_nbrs_per_node[p_id]) <= 2:
            continue
          start = cumm_sampled_nbrs_per_node[p_id][p_counters[p_id]]
          p_counters[p_id] += 1
          end = cumm_sampled_nbrs_per_node[p_id][p_counters[p_id]]

          node_with_dupl = torch.cat([node_with_dupl, results[p_id].node[start: end]])
          edge = torch.cat([edge, results[p_id].edge[start: end]]) if self.with_edge else None
          batch = torch.cat([batch, results[p_id].batch[start: end]]) if self.disjoint else None

          sampled_nbrs_per_node += [end - start]

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
    one_hop_num: Union[int, Dict[EdgeType, List[int]]],
    seed_time: Optional[Union[Tensor, Dict[NodeType, Tensor]]] = None,
    batch: OptTensor = None,
    edge_type: Optional[EdgeType] = None,
  ) -> Union[SamplerOutput, HeteroSamplerOutput]:
    r""" Sample one-hop neighbors and induce the coo format subgraph.

    Args:
      srcs: input ids, 1D tensor.
      num_nbr: request(max) number of neighbors for one hop.
      etype: edge type to sample from input ids.

    Returns:
      Tuple[torch.Tensor, torch.Tensor]: unique node ids and edge_index.
    """
    device = torch.device(type='cpu')

    srcs = srcs.to(device)
    batch = batch.to(device) if batch is not None else None
    seed_time = seed_time.to(device) if seed_time is not None else None

    nodes = torch.arange(srcs.size(0), dtype=torch.long, device=device)
    src_ntype = edge_type[2] if edge_type is not None else None # if csc
    
    partition_ids = self.dist_graph.get_partition_ids_from_nids(srcs, src_ntype)
    partition_ids = partition_ids.to(self.device)

    partition_results: List[Union[SamplerOutput, HeteroSamplerOutput]] = [None] * self.dist_graph.meta['num_parts']
    remote_nodes: List[torch.Tensor] = []
    futs: List[torch.futures.Future] = []

    for i in range(self.dist_graph.num_partitions):
      p_id = (
        (self.dist_graph.partition_idx + i) % self.dist_graph.num_partitions
        #(self.data.partition_idx + i) % self.data.num_partitions
      )
      p_mask = (partition_ids == p_id)
      p_srcs = torch.masked_select(srcs, p_mask)
      p_batch = torch.masked_select(batch, p_mask) if batch is not None else None
      p_seed_time = torch.masked_select(seed_time, p_mask) if seed_time is not None else None

      if p_srcs.shape[0] > 0:
        p_nodes = torch.masked_select(nodes, p_mask)
        if p_id == self.dist_graph.partition_idx:
          
          p_nbr_out = self._sampler.sample_one_hop(p_srcs, one_hop_num, p_seed_time, p_batch, edge_type)
          partition_results.pop(p_id)
          partition_results.insert(p_id, p_nbr_out)
        else:
          remote_nodes.append(p_nodes)
          to_worker = self.rpc_router.get_to_worker(p_id)
          futs.append(rpc_async(to_worker,
                                        self.rpc_sample_callee_id,
                                        args=(p_srcs.cpu(), one_hop_num, p_seed_time, p_batch.cpu() if p_batch is not None else None, edge_type)))

    # Without remote sampling results.
    if len(remote_nodes) == 0:
      return self.merge_results(partition_ids, partition_results, edge_type)
    # With remote sampling results.
    res_fut_list = await wrap_torch_future(torch.futures.collect_all(futs))
    for i, res_fut in enumerate(res_fut_list):
      
      partition_results.pop(p_id)
      partition_results.insert(p_id, res_fut.wait())
    
    return self.merge_results(partition_ids, partition_results, edge_type)

  async def _colloate_fn(
    self,
    output: Union[SamplerOutput, HeteroSamplerOutput]
  ) -> SamplerOutput:
    r""" Collect labels and features for the sampled subgrarph if necessary,
    and put them into a sample message.
    """
    result_map = {}
    # if isinstance(output.metadata, dict):
      #scan kv and add metadata
    if isinstance(output.metadata, dict):
        input_type = output.metadata.get('input_type', '')
    else:
        # ! Workaround for mismatch in metadata type here: dict() PyG NeighborSampler: tuple()
        input_type = None
        # erasing all metadata here
        output.metadata = {'og_meta': output.metadata}
    # print(f"input_type from output.metadata={input_type}")
      # batch_size = output.metadata.get('bs', 1)
      # result_map['meta'] = torch.LongTensor([int(is_hetero), batch_size])
      # # output.metadata.pop('input_type', '')
      # # output.metadata.pop('bs', 1)
      # for k, v in output.metadata.items():
      #   result_map[k] = v
  
    if self.dist_graph.meta["is_hetero"]:
      for ntype, nodes in output.node.items():
        result_map[f'{as_str(ntype)}.ids'] = nodes
      for etype, rows in output.row.items():
        etype_str = as_str(etype)
        result_map[f'{etype_str}.rows'] = rows
        result_map[f'{etype_str}.cols'] = output.col[etype]
        if self.with_edge:
          result_map[f'{etype_str}.eids'] = output.edge[etype]
          
      # Collect node labels of input node type.
      if not isinstance(input_type, Tuple):
        node_labels = self.data.get_node_label(input_type)
        if node_labels is not None:
          result_map[f'{as_str(input_type)}.nlabels'] = \
            node_labels[output.node[input_type]]
      # Collect node features.
      if self.dist_node_feature is not None:
        nfeat_fut_dict = {}
        for ntype, nodes in output.node.items():
          nodes = nodes.to(torch.long)
          nfeat_fut_dict[ntype] = self.dist_node_feature.async_get(nodes, ntype)
        for ntype, fut in nfeat_fut_dict.items():
          nfeats = await wrap_torch_future(fut)
          result_map[f'{as_str(ntype)}.nfeats'] = nfeats
      # Collect edge features
      if self.dist_edge_feature is not None and self.with_edge:
        efeat_fut_dict = {}
        for etype in self.edge_types:
          eids = result_map.get(f'{as_str(etype)}.eids', None).to(torch.long)
          if eids is not None:
            efeat_fut_dict[etype] = self.dist_edge_feature.async_get(eids, etype)
        for etype, fut in efeat_fut_dict.items():
          efeats = await wrap_torch_future(fut)
          result_map[f'{as_str(etype)}.efeats'] = efeats
    else:
      # result_map['ids'] = output.node
      # result_map['rows'] = output.row
      # result_map['cols'] = output.col
      # if self.with_edge:
      #   result_map['eids'] = output.edge
      # Collect node labels.
      node_labels = self.dist_graph.labels #self.data.get_node_label()
      if node_labels is not None:
        #result_map['nlabels'] = node_labels[output.node]
        output.metadata['nlabels'] = node_labels[output.node]
      # Collect node features.
      if self.dist_node_feature is not None:
        #print(f"------- 777.3---- DistNSampler: _colloate_fn()   dist_node_feature.async_get(),   output.node={output.node}, self.dist_node_feature={self.dist_node_feature}, self.dist_edge_feature={self.dist_edge_feature} -------")
        fut = self.dist_node_feature.lookup_features(is_node_feat=True, ids=output.node)
        #print(f"Sampler PID-{mp.current_process().pid} 1: async_get returned {fut}")
        nfeats = await wrap_torch_future(fut) #torch.Tensor([])
        #print(f"Sampler PID-{mp.current_process().pid} 2: wrap_torch_feature returned {nfeats}")
        #result_map['nfeats'] = nfeats
        output.metadata['nfeats'] = nfeats.to(torch.device('cpu'))
        output.edge=torch.empty(0)

      # Collect edge features.
      if self.dist_edge_feature is not None:
        eids = result_map['eids']
        fut = self.dist_edge_feature.lookup_features(is_node_feat=False, ids=eids)
        efeats = await wrap_torch_future(fut)
        output.metadata['efeats'] = efeats
      else:
        output.metadata['efeats'] = None
      
      #print(f"------- 777.4 ----- DistNSampler: _colloate_fn()  return -------")

    return output #result_map

  def __repr__(self):
    return f"{self.__class__.__name__}()-PID{mp.current_process().pid}"
  
# Sampling Utilities ##########################################################

def close_sampler(worker_id, sampler):
  print(f"Closing rpc in {repr(sampler)} worker-id {worker_id}")
  try:
    sampler.event_loop.shutdown_loop()
  except AttributeError:
    pass
  shutdown_rpc(graceful=True)
