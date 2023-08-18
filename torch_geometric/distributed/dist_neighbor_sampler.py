import torch.multiprocessing as mp
import torch

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
from .pyg_messagequeue import PyGMessageQueue

#from .dist_dataset import DistDataset
#from .dist_feature import DistFeature
#from .dist_graph import DistGraph

r"""
from .event_loop import ConcurrentEventLoop, wrap_torch_future
from .rpc import (
  RpcCallBase, rpc_register, rpc_request_async,
  RpcRouter, rpc_partition2workers, shutdown_rpc
)
##
from torch_geometric.data import Data
from torch_geometric.data import TensorAttr
"""
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

r"""
@dataclass
class PartialNeighborOutput:
  index: torch.Tensor
  output: SamplerOutput


class RpcSamplingCallee(RpcCallBase):
  def __init__(self, sampler: NeighborSampler, device: torch.device):
    super().__init__()
    self.sampler = sampler
    self.device = device

  def rpc_async(self, *args, **kwargs):
    ensure_device(self.device)
    output = self.sampler.sample_one_hop(*args, **kwargs)

    #if(output.device=='cpu'):
    return output
    #else:
    #    return output.to(torch.device('cpu'))
    
  def rpc_sync(self, *args, **kwargs):
      pass

class RpcSubGraphCallee(RpcCallBase):
  def __init__(self, sampler: NeighborSampler, device: torch.device):
    super().__init__()
    self.sampler = sampler
    self.device = device

  def rpc_async(self, *args, **kwargs):
    ensure_device(self.device)
    with_edge = kwargs['with_edge']
    output = self.sampler.subgraph_op.node_subgraph(args[0].to(self.device),
                                                    with_edge)
    eids = output.eids.to('cpu') if with_edge else None
    return output.nodes.to('cpu'), output.rows.to('cpu'), output.cols.to('cpu'), eids

  def rpc_sync(self, *args, **kwargs):
      pass
"""

@dataclass
class PartialNeighborOutput:
  r""" The sampled neighbor output of a subset of the original ids.

  * index: the index of the subset vertex ids.
  * output: the sampled neighbor output.
  """
  index: torch.Tensor
  output: SamplerOutput


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
    
    self.with_edge = True
    self.with_node = True



  def register_sampler_rpc(self):  
    
    partition2workers = rpc_partition_to_workers(
    current_ctx = self.current_ctx,
    num_partitions=self.dist_graph.num_partitions,
    current_partition_idx=self.dist_graph.partition_idx
    )
    
    self.rpc_router = RPCRouter(partition2workers)
    
    # collect node\edge features
    try:
      node_features = self.dist_feature.get_tensor(group_name=None, attr_name='x')
      print(f"--000000000000.1-- node_features={node_features} ")
    except KeyError:
      self.with_node = False
    
    try:
      edge_features = self.dist_feature.get_tensor(group_name=(None,None), attr_name='edge_attr')
      print(f"--000000000000.2-- edge_features={edge_features} ")
    except KeyError:
      self.with_edge = False

    if any((self.with_node, self.with_edge)):
        self.dist_feature.set_rpc_router(self.rpc_router)
    else:
      raise AttributeError("Provided LocalFeatureStore doesn't contain any node\edge features.")
        
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

    if self.dist_graph.meta["is_hetero"]:
      self.num_neighbors = self._sampler.num_neighbors
      self.num_hops = self._sampler.num_hops
      self.edge_types = self._sampler.edge_types

    #self.sample_fn = None
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
      return self.event_loop.run_task(coro=self._send_adapter(self.node_sample,
                                                   inputs))
    cb = kwargs.get('callback', None)
    self.event_loop.add_task(coro=self._send_adapter(self.node_sample, inputs),
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

#   def subgraph(
#     self,
#     inputs: NodeSamplerInput,
#     **kwargs
#   ) ->  Union[SamplerOutput, HeteroSamplerOutput]:
#     r""" Induce an enclosing subgraph based on inputs and their neighbors(if
#       self.num_neighbors is not None).
#     """
#     inputs = NodeSamplerInput.cast(inputs)
#     if self.channel is None:
#       return self.run_task(coro=self._send_adapter(self._subgraph, inputs))
#     cb = kwargs.get('callback', None)
#     self.add_task(coro=self._send_adapter(self._subgraph, inputs), callback=cb)
#     return None

  async def _send_adapter(
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
    
    #print(f" ----777.1 -------- distNSampler:  node_sample, inputs={inputs}, seed={seed}, input_type={input_type} ")
    if(self.dist_graph.meta["is_hetero"]):
      assert input_type is not None
      src_dict = inducer.init_node({input_type: input_seeds})
      batch_size = src_dict[input_type].numel()

      out_nodes, out_rows, out_cols, out_edges = {}, {}, {}, {}
      merge_dict(src_dict, out_nodes)

      for i in range(self.num_hops):
        task_dict, nbr_dict, edge_dict = {}, {}, {}
        for etype in self.edge_types:
          srcs = src_dict.get(etype[0], None)
          req_num = self.num_neighbors[etype][i]
          if srcs is not None:
            task_dict[etype] = self._loop.create_task(
              self._sample_one_hop(inputs, req_num, etype))
        for etype, task in task_dict.items():
          output: NeighborOutput = await task
          nbr_dict[etype] = [src_dict[etype[0]], output.nbr, output.nbr_num]
          if output.edge is not None:
            edge_dict[etype] = output.edge
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
        metadata=metadata
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
        metadata=metadata
       )

    return sample_output

#   async def _subgraph(
#     self,
#     inputs: NodeSamplerInput,
#   ) -> Optional[Dict[str, torch.Tensor]]:
#     #TODO: refactor
#     inputs = NodeSamplerInput.cast(inputs)
#     input_seeds = inputs.node.to(self.device)
#     #is_hetero = (self.dist_graph.data_cls == 'hetero')
#     #if is_hetero:
#     if self.dist_graph.meta["is_hetero"]:
#       raise NotImplementedError
#     else:
#       # neighbor sampling.
#       if self.num_neighbors is not None:
#         nodes = [input_seeds]
#         for num in self.num_neighbors:
#           nbr = await self._sample_one_hop(nodes[-1], num)
#           nodes.append(torch.unique(nbr.nbr))
#         nodes = torch.cat(nodes)
#       else:
#         nodes = input_seeds
#       nodes, mapping = torch.unique(nodes, return_inverse=True)
#       nid2idx = id2idx(nodes)
#       # subgraph inducing.
#       partition_ids = self.dist_graph.get_node_partitions(nodes)
#       partition_ids = partition_ids.to(self.device)
#       rows, cols, eids, futs = [], [], [], []
#       for i in range(self.data.num_partitions):
#         p_id = (self.data.partition_idx + i) % self.data.num_partitions
#         p_ids = torch.masked_select(nodes, (partition_ids == p_id))
#         if p_ids.shape[0] > 0:
#           if p_id == self.data.partition_idx:
#             subgraph = self._sampler.subgraph_op.node_subgraph(nodes, self.with_edge)
#             # relabel row and col indices.
#             rows.append(nid2idx[subgraph.nodes[subgraph.rows]])
#             cols.append(nid2idx[subgraph.nodes[subgraph.cols]])
#             if self.with_edge:
#               eids.append(subgraph.eids.to(self.device))
#           else:
#             to_worker = self.rpc_router.get_to_worker(p_id)
#             futs.append(rpc_async(to_worker,
#                                           self.rpc_subgraph_callee_id,
#                                           args=(nodes.cpu(),),
#                                           kwargs={'with_edge': self.with_edge}))
#       if not len(futs) == 0:
#         res_fut_list = await wrap_torch_future(torch.futures.collect_all(futs))
#         for res_fut in res_fut_list:
#           res_nodes, res_rows, res_cols, res_eids = res_fut.wait()
#           res_nodes = res_nodes.to(self.device)
#           rows.append(nid2idx[res_nodes[res_rows]])
#           cols.append(nid2idx[res_nodes[res_cols]])
#           if self.with_edge:
#             eids.append(res_eids.to(self.device))

#       sample_output = SamplerOutput(
#         node=nodes,
#         row=torch.cat(rows),
#         col=torch.cat(cols),
#         edge=torch.cat(eids) if self.with_edge else None,
#         device=self.device,
#         metadata={'mapping': mapping[:input_seeds.numel()]})

#       return sample_output

  def merge_results(
    self,
    partition_ids: torch.Tensor,
    results: List[PartialNeighborOutput]
  ) -> NeighborOutput:
    r""" Merge partitioned neighbor outputs into a complete one.
    """
    partition_ids = partition_ids.tolist()
    cumm_sampled_nbrs_per_node = [r.output.metadata if r is not None else None for r in results]
    p_counters = [0] * self.dist_graph.meta['num_parts']

    node_with_dupl  = torch.empty(0, dtype=torch.int64)
    edge = torch.empty(0, dtype=torch.int64) if self.with_edge else None
    batch = torch.empty(0, dtype=torch.int64) if self.disjoint else None
    sampled_nbrs_per_node = []

    #print(f"--------YYY.0   --- partition_ids={partition_ids}, cumm_sampled_nbrs_per_node={cumm_sampled_nbrs_per_node} ")

    for p_id in partition_ids:
      #print(f"----------YYY.1   p_id={p_id} cumm_sampled_nbrs_per_node={cumm_sampled_nbrs_per_node[p_id]} ")
        if cumm_sampled_nbrs_per_node[p_id] is None:
          continue
        elif len(cumm_sampled_nbrs_per_node[p_id]) < 2:
          continue
        else:
          start = cumm_sampled_nbrs_per_node[p_id][p_counters[p_id]]
          p_counters[p_id] += 1
          end = cumm_sampled_nbrs_per_node[p_id][p_counters[p_id]]

          node_with_dupl = torch.cat([node_with_dupl, results[p_id].output.node[start: end]])
          edge = torch.cat([edge, results[p_id].output.edge[start: end]]) if self.with_edge else None
          batch = torch.cat([batch, results[p_id].output.batch[start: end]]) if self.disjoint else None

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
    srcs: torch.Tensor,
    one_hop_num: int,
    seed_time: OptTensor = None,
    batch: OptTensor = None,
    src_etype: Optional[EdgeType] = None,
  ) -> SamplerOutput:
  #) -> NeighborOutput:
    r""" Sample one-hop neighbors and induce the coo format subgraph.

    Args:
      srcs: input ids, 1D tensor.
      num_nbr: request(max) number of neighbors for one hop.
      etype: edge type to sample from input ids.

    Returns:
      Tuple[torch.Tensor, torch.Tensor]: unique node ids and edge_index.
    """
    device = torch.device(type='cpu')

    if self.dist_graph.partition_idx==0:
        COLOR = '\33[32m'
    else:
        COLOR = '\33[37m'
    CEND = '\033[0m'

    print(COLOR + f"\n\n\n\n XXX.0--PID-{mp.current_process().pid}---- DistNSampler: async _sample_one_hop(), one_hop_num={one_hop_num}, pidx={self.dist_graph.partition_idx}, srcs.size(0)={srcs.size(0)}, srcs={srcs}, batch={batch}, seed_time={seed_time} ------"+ CEND)

    srcs = srcs.to(device) # all src nodes
    batch = batch.to(device) if batch is not None else None
    seed_time = seed_time.to(device) if seed_time is not None else None

    
    nodes = torch.arange(srcs.size(0), dtype=torch.long, device=device)
    src_ntype = src_etype[0] if src_etype is not None else None
    
    partition_ids = self.dist_graph.get_partition_ids_from_nids(srcs, src_ntype)
    partition_ids = partition_ids.to(self.device)

    #print(COLOR + f"\n\n\n\n COLOR + XXX.1--PID-{mp.current_process().pid}---- DistNSampler: async _sample_one_hop(), one_hop_num={one_hop_num}, pidx={self.dist_graph.partition_idx}, partition_ids={partition_ids} ------"+ CEND)
    
    partition_results: List[PartialNeighborOutput] = [None] * self.dist_graph.meta['num_parts']
    remote_nodes: List[torch.Tensor] = []
    futs: List[torch.futures.Future] = []


    #print(f"\n\n\n-------- DistNSample -------------------------------------\n\n\n")

    for i in range(self.dist_graph.num_partitions):
      p_id = (
        (self.dist_graph.partition_idx + i) % self.dist_graph.num_partitions
      )
      p_mask = (partition_ids == p_id)
      p_srcs = torch.masked_select(srcs, p_mask)
      p_batch = torch.masked_select(batch, p_mask) if batch is not None else None
      p_seed_time = torch.masked_select(seed_time, p_mask) if seed_time is not None else None

      if p_srcs.shape[0] > 0:
        p_nodes = torch.masked_select(nodes, p_mask)
        if remote_nodes is None:
          print("-----11--------#########################-------- remote_nodes is None")
        if len(p_nodes) == 0:
          print("-----22--------#########################-------- remote_nodes is None")

        if p_id == self.dist_graph.partition_idx:
      
          print(COLOR + f"----000--PID-{mp.current_process().pid}--one_hop_num={one_hop_num}-----, _sample_one_hop(): Node{self.dist_graph.partition_idx}-LOCAL---------, pidx={self.dist_graph.partition_idx}, p_id={p_id}, p_nodes.shape={p_nodes.shape}, p_nodes={p_nodes} ------" + CEND)
          p_nbr_out = self._sampler.sample_one_hop(p_srcs, one_hop_num, p_seed_time, p_batch, src_etype)
          print(COLOR + f"----000.2--PID-{mp.current_process().pid}--one_hop_num={one_hop_num}----- _sample_one_hop(): Node{self.dist_graph.partition_idx}-LOCAL---------, pidx={self.dist_graph.partition_idx}, p_nbr_out={p_nbr_out}, node.size={p_nbr_out.node.size()}, edge.size=={p_nbr_out.edge.size()}, meta_len ={len(p_nbr_out.metadata)}----\n\n"+ CEND)
          partition_results.pop(p_id)
          partition_results.insert(p_id, PartialNeighborOutput(p_nodes, p_nbr_out))
        else:
          print(COLOR + f"----111--PID-{mp.current_process().pid}--one_hop_num={one_hop_num}----- _sample_one_hop(): Node{self.dist_graph.partition_idx}- REMOTE --------, pidx={self.dist_graph.partition_idx}, p_id={p_id}, p_nodes.shape={p_nodes.shape}, p_nodes={p_nodes}  ------" + CEND)
          remote_nodes.append(p_nodes)
          to_worker = self.rpc_router.get_to_worker(p_id)
          
          #print(f"----111.1---- DistNSampler: async _sample_one_hop(), to_worker={to_worker} ------")
          temp_futs = rpc_async(to_worker,
                                        self.rpc_sample_callee_id,
                                        args=(p_srcs.cpu(), one_hop_num, p_seed_time, p_batch.cpu() if p_batch is not None else None, src_etype))
          print(COLOR + f"----111.2--PID-{mp.current_process().pid}--one_hop_num={one_hop_num}----- _sample_one_hop(): Node{self.dist_graph.partition_idx}- REMOTE --------, pidx={self.dist_graph.partition_idx}, temp_futs={temp_futs.wait()}, node.size={temp_futs.wait().node.size()}, edge.size=={temp_futs.wait().edge.size()}, meta_len ={len(temp_futs.wait().metadata)} ------\n\n"+ CEND)
          futs.append(temp_futs)

    #print(f"-----333--- DistNSampler: async _sample_one_hop() without remote sampling results -------")
    # Without remote sampling results.
    if remote_nodes is None:
      print("------3333333333-------#########################-------- remote_nodes is None")

    if len(remote_nodes) == 0:
      print(COLOR + f"--------NO REMOTE NEIGHBORS !!!!!!! --PID-{mp.current_process().pid}----one_hop_num={one_hop_num}, pidx={self.dist_graph.partition_idx}, partition_results ={partition_results}---  "+ CEND)
      #print(f"--------NO REMOTE NEIGHBORS !!!!!!! ----one_hop_num={one_hop_num}, pidx={self.dist_graph.partition_idx}, partition_results[self.dist_graph.partition_idx].output ={partition_results[self.dist_graph.partition_idx].output}---  ")
      return self.merge_results(partition_ids, partition_results)
      #return partition_results[self.dist_graph.partition_idx].output
    # With remote sampling results.
    res_fut_list = await wrap_torch_future(torch.futures.collect_all(futs))
    for i, res_fut in enumerate(res_fut_list):
      #print(f"---444----- DistNSampler: async _sample_one_hop() res_fut={res_fut.wait()} -------")
      
      partition_results.pop(p_id)
      partition_results.insert(p_id,
        PartialNeighborOutput(
          index=remote_nodes[i],
          output=res_fut.wait()
        )
      )
    
    print(COLOR + f"\n\n-------55555--PID-{mp.current_process().pid}---- one_hop_num={one_hop_num}, pidx={self.dist_graph.partition_idx}, -- DistNSampler: _sample_one_hop() before stitching -----------------  partition_results={partition_results}-------"+ CEND)
    #print(f"-------- DistNSampler: async _sample_one_hop() before stitching -----------------  partition_results={partition_results[0].output.node.size()}, {partition_results[1].output.node.size()}-------")
    #print("\n\n\n\n")
    return self.merge_results(partition_ids, partition_results)

  async def _colloate_fn(
    self,
    output: Union[SamplerOutput, HeteroSamplerOutput]
  ) -> SamplerOutput:
    r""" Collect labels and features for the sampled subgrarph if necessary,
    and put them into a sample message.
    """
    result_map = {}

    input_type = output.metadata[2]
    
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
      if self.dist_feature is not None:
        nfeat_fut_dict = {}
        for ntype, nodes in output.node.items():
          nodes = nodes.to(torch.long)
          nfeat_fut_dict[ntype] = self.dist_feature.async_get(nodes, ntype)
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
        # Collect node labels.        
        nlabels = self.dist_graph.labels[output.node] if (self.dist_graph.labels is not None) else None
        # Collect node features.
        if self.with_node:
          fut = self.dist_feature.lookup_features(is_node_feat=True, ids=output.node)
          nfeats = await wrap_torch_future(fut) 
          nfeats = nfeats.to(torch.device('cpu'))
        else:
          nfeats = None
        # Collect edge features.
        if self.with_edge and output.edge is not None:
          fut = self.dist_feature.lookup_features(is_node_feat=False, ids=output.edge)
          efeats = await wrap_torch_future(fut)
          efeats = efeats.to(torch.device('cpu'))
        else: 
          efeats = None

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
  
