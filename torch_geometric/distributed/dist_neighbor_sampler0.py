import math
import queue
import multiprocessing as mp

from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable

import torch

import atexit

#from ..channel import ChannelBase, SampleMessage
from ..sampler import (
  NodeSamplerInput, EdgeSamplerInput,
  NeighborOutput, SamplerOutput, HeteroSamplerOutput,
  NeighborSampler
)
from ..typing import EdgeType, as_str, NumNeighbors
from ..utils import (
    #get_available_device, 
    ensure_device, 
    merge_dict, id2idx,
    merge_hetero_sampler_output, format_hetero_sampler_output,
    id2idx_v2
)

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

@dataclass
class PartialNeighborOutput:
  r""" The sampled neighbor output of a subset of the original ids.

  * index: the index of the subset vertex ids.
  * output: the sampled neighbor output.
  """
  index: torch.Tensor
  output: NeighborOutput


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
               disjoint: bool = False,
               with_edge: bool = False,
               with_neg: bool = False,
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
    self.max_input_size = 0
    self.with_edge = with_edge
    self.with_neg = with_neg
    self.collect_features = collect_features
    self.channel = channel
    self.concurrency = concurrency
    self.device = device #get_available_device(device)
    self.event_loop = None
    self.replace = replace
    self.subgraph_type = subgraph_type
    self.disjoint = disjoint
    self.temporal_strategy = kwargs.pop('temporal_strategy', 'uniform')
    self.time_attr = kwargs.pop('time_attr', None)
    print(f"---- 555.2 -------- dist_neighbor_sampler - init done, self.current_context={self.current_ctx}  ")

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
      
      edge_index = self.graph.get_edge_index(edge_type=None, layout='coo')
      print(f"----////////88888//////////---- edge_index={edge_index}    ")

      self.dist_node_feature = None
      self.dist_edge_feature = None
      if self.collect_features:
        node_features = self.feature.get_tensor(group_name=None, attr_name='x')
        print(f"--000000000000.1-- node_features={node_features} ")
      
        if node_features is not None:
            local_feature=self.feature
            local_feature.set_rpc_router(self.rpc_router)
            
            self.dist_node_feature = local_feature

        attrs = self.feature.get_all_tensor_attrs()
        print(f"------------- attrs={attrs}  ")

        edge_features = None #data[4].get_tensor(group_name=attrs[0].group_name, attr_name=attrs[0].attr_name)
        #group_name=(None, None), attr_name='edge_attr'
        #print(f"--000000000000.2-- edge_features={edge_features} ")

        if self.with_edge and edge_features is not None:
          self.dist_edge_feature = None
          r""" DistFeature(meta=data[0],num_partitions=data[1], partition_index=data[2],local_feature=data[4], feature_pb=data[6], local_only=False, rpc_router=self.rpc_router, device=self.device)"""
        r"""
        if data.node_features is not None:
          self.dist_node_feature = DistFeature(data.meta,
            data.num_partitions, data.partition_idx,
            data.node_features, data.node_feat_pb,
            local_only=False, rpc_router=self.rpc_router, device=self.device
          )
        if self.with_edge and data.edge_features is not None:
          self.dist_edge_feature = DistFeature(data.meta,
            data.num_partitions, data.partition_idx,
            data.edge_features, data.edge_feat_pb,
            local_only=False, rpc_router=self.rpc_router, device=self.device
          )
        """
    else:
      raise ValueError(f"'{self.__class__.__name__}': found invalid input "
                       f"data type '{type(data)}'")

    print(f"---- 666.2 -------- register_rpc done    ")


    data0 = Data(x= node_features, edge_index=edge_index)
    print(f"-----------  data0={data0} ")

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

    if self.dist_graph.meta["is_hetero"]:
      self.num_neighbors = self._sampler.num_neighbors
      self.num_hops = self._sampler.num_hops
      self.edge_types = self._sampler.edge_types

    #self.sample_fn = None
    print(f"---- 666.3 -------- register_rpc done    ")

  
  
  def init_event_loop(self):

    print(f"---- 777.1 -------- init_event_loop    ")
    self.event_loop = ConcurrentEventLoop(self.concurrency)
    self.event_loop._loop.call_soon_threadsafe(ensure_device, self.device)
    self.event_loop.start_loop()
    print(f"----------- {repr(self)}:init_event_loop()   END ------------- ")

  def set_sample_fn(self, sampling_type):
    if sampling_type == SamplingType.NODE:
      self.sample_fn = self.sample_from_nodes
    elif sampling_type == SamplingType.LINK:
      self.sample_fn = self.sample_from_edges
    elif sampling_type == SamplingType.SUBGRAPH:
      self.sample_fn = self.subgraph
    else:
      raise NotImplementedError

  def __repr__(self):
    return f"{self.__class__.__name__}()-PID{mp.current_process().pid}"

  def _sample(
    self,
    inputs: Union[NodeSamplerInput,EdgeSamplerInput],
    _sample_fn: Callable,
    **kwargs
  ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:

    if self.channel is None:
      return self.event_loop.run_task(coro=self._send_adapter(_sample_fn, inputs))

    cb = kwargs.get('callback', None)
    self.event_loop.add_task(coro=self._send_adapter(_sample_fn, inputs), callback=cb)
    return None



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
    inputs = NodeSamplerInput.cast(inputs)
    out = sample(inputs, self._sample, self._sample_from_nodes)
    return out

  def sample_from_edges(
    self,
    inputs: EdgeSamplerInput,
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

    #inputs = NodeSamplerInput.cast(inputs)
    out = sample(inputs, self._sample, self._sample_from_edges)
    return out


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
  ) -> Optional[Dict[str, torch.Tensor]]:
    sampler_output = await async_func(*args, **kwargs)
    res = await self._colloate_fn(sampler_output)

    #print(f"\n\n--------------   dist_neighbor_sampler:   res={res} ----------- \n\n")
    #torch.save(res, "sample_output_for_channel.pt")

    if self.channel is None:
      return res
    self.channel.put(res)
    return None

  async def _sample_from_nodes(
    self,
    inputs: NodeSamplerInput,
  ) -> Optional[Dict[str, torch.Tensor]]:
    input_seeds = inputs.node.to(self.device)
    input_type = inputs.input_type
    self.max_input_size = max(self.max_input_size, input_seeds.numel())
    ##inducer = self._acquire_inducer()
    #is_hetero = self.dist_graph.is_hetero # (self.dist_graph.data_cls == 'hetero')
    
    #*print(f" ----777.1 -------- distNSampler:  _sample_from_nodes, self.dist_graph.data_cls={self.dist_graph.data_cls}, input_seeds={input_seeds}, input_type={input_type} ")
    #if is_hetero:
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
              self._sample_one_hop(srcs, req_num, etype))
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
        metadata={'input_type': input_type, 'bs': batch_size}
      )
    else:
      ##srcs = inducer.init_node(input_seeds)
      srcs = input_seeds
      batch_size = srcs.numel()

      out_nodes, out_edges = [], []

      device = torch.device(type='cpu')

      #out_nodes.to(device)
      #out_edges.to(device)

      out_nodes.append(srcs.to(device))
      # Sample subgraph.

      #*print(f" ----777.11-------- distNSampler:  out_nodes={ out_nodes}, self.num_neigyyhbors={self.num_neighbors} ")

      for req_num in self.num_neighbors:
        output = await self._sample_one_hop(srcs, req_num, None)
        #*print(f" ----777.12----- distNSampler:  output={output} , req_num={req_num}")
        #*print("\n")


        out_nodes.append(output.node)
        out_edges.append((output.row, output.col, output.edge))
        srcs = output.node

        r"""
        nodes, rows, cols = \
          inducer.induce_next(srcs, output.nbr, output.nbr_num)
        out_nodes.append(nodes)
        out_edges.append((rows, cols, output.edge))
        srcs = nodes
        """

      sample_output = SamplerOutput(
        node=torch.cat(out_nodes),
        row=torch.cat([e[0] for e in out_edges]),
        col=torch.cat([e[1] for e in out_edges]),
        edge=(torch.cat([e[2] for e in out_edges]) if self.with_edge else None),
        metadata={'input_type': None, 'bs': batch_size}
      )
    # Reclaim inducer into pool.
    #self.inducer_pool.put(inducer)

    #*print(f"----777.2-------- 555 _sampler_from_nodes Done -----------")
    #*print("\n\n\n\n")
    #return sample_output
    return sample_output


  async def _sample_from_edges(
      self,
      inputs: EdgeSamplerInput,
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
    src = inputs.row.to(self.device)
    dst = inputs.col.to(self.device)
    edge_label = None if inputs.label is None else inputs.label.to(self.device)
    input_type = inputs.input_type
    neg_sampling = inputs.neg_sampling

    num_pos = src.numel()
    num_neg = 0
    # Negative Sampling
    self._sampler.lazy_init_neg_sampler()
    if neg_sampling is not None:
      # When we are doing negative sampling, we append negative information
      # of nodes/edges to `src`, `dst`.
      # Later on, we can easily reconstruct what belongs to positive and
      # negative examples by slicing via `num_pos`.
      num_neg = math.ceil(num_pos * neg_sampling.amount)
      if neg_sampling.is_binary():
        # In the "binary" case, we randomly sample negative pairs of nodes.
        if input_type is not None:
          neg_pair = self._sampler._neg_sampler[input_type].sample(num_neg)
        else:
          neg_pair = self._sampler._neg_sampler.sample(num_neg)
        src_neg, dst_neg = neg_pair[0], neg_pair[1]
        src = torch.cat([src, src_neg], dim=0)
        dst = torch.cat([dst, dst_neg], dim=0)
        if edge_label is None:
            edge_label = torch.ones(num_pos, device=self.device)
        size = (num_neg, ) + edge_label.size()[1:]
        edge_neg_label = edge_label.new_zeros(size)
        edge_label = torch.cat([edge_label, edge_neg_label])
      elif neg_sampling.is_triplet():
        assert num_neg % num_pos == 0
        if input_type is not None:
          neg_pair = self._sampler._neg_sampler[input_type].sample(num_neg, padding=True)
        else:
          neg_pair = self._sampler._neg_sampler.sample(num_neg, padding=True)
        dst_neg = neg_pair[1]
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
        temp_out.append(await self._sample_from_nodes(seeds))
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
      out = await self._sample_from_nodes(NodeSamplerInput.cast(seed))

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
        out.metadata.update({'src_index': src_index,
                             'dst_pos_index': dst_pos_index,
                             'dst_neg_index': dst_neg_index})

    return out


  async def _subgraph(
    self,
    inputs: NodeSamplerInput,
  ) -> Optional[Dict[str, torch.Tensor]]:
    inputs = NodeSamplerInput.cast(inputs)
    input_seeds = inputs.node.to(self.device)
    #is_hetero = self.dist_graph.is_hetero #(self.dist_graph.data_cls == 'hetero')
    #if is_hetero:
    if(self.dist_graph.meta["is_hetero"]):
      raise NotImplementedError
    else:
      # neighbor sampling.
      if self.num_neighbors is not None:
        nodes = [input_seeds]
        for num in self.num_neighbors:
          nbr = await self._sample_one_hop(nodes[-1], num, None)
          nodes.append(torch.unique(nbr.nbr))
        nodes = torch.cat(nodes)
      else:
        nodes = input_seeds
      nodes, mapping = torch.unique(nodes, return_inverse=True)
      nid2idx = id2idx(nodes)
      # subgraph inducing.
      partition_ids = self.dist_graph.get_partition_ids_from_nids(nodes)
      partition_ids = partition_ids.to(self.device)
      rows, cols, eids, futs = [], [], [], []
      for i in range(self.data.num_partitions):
        pidx = (self.data[2] + i) % self.data.num_partitions
        p_ids = torch.masked_select(nodes, (partition_ids == pidx))
        if p_ids.shape[0] > 0:
          if pidx == self.data[2]:
            subgraph = self._sampler.subgraph_op.node_subgraph(nodes, self.with_edge)
            # relabel row and col indices.
            rows.append(nid2idx[subgraph.nodes[subgraph.rows]])
            cols.append(nid2idx[subgraph.nodes[subgraph.cols]])
            if self.with_edge:
              eids.append(subgraph.eids.to(self.device))
          else:
            to_worker = self.rpc_router.get_to_worker(pidx)
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

  def _acquire_inducer(self):
    if self.inducer_pool.empty():
      return self._sampler.create_inducer(self.max_input_size)
    return self.inducer_pool.get()

  def _stitch_sample_results(
    self,
    input_seeds: torch.Tensor,
    results: List[PartialNeighborOutput]
  ) -> NeighborOutput:
    r""" Stitch partitioned neighbor outputs into a complete one.
    """

    r"""
    idx_list = [r.index for r in results]
    nbrs_list = [r.output.nbr for r in results]
    nbrs_num_list = [r.output.nbr_num for r in results]
    eids_list = [r.output.edge for r in results] if self.with_edge else []

    if self.device.type == 'cuda':
      nbrs, nbrs_num, eids = pywrap.cuda_stitch_sample_results(
        input_seeds, idx_list, nbrs_list, nbrs_num_list, eids_list)
    else:
      nbrs, nbrs_num, eids = pywrap.cpu_stitch_sample_results(
        input_seeds, idx_list, nbrs_list, nbrs_num_list, eids_list)
    #return NeighborOutput(nbrs, nbrs_num, eids)
    """
    idx_list = torch.cat([r.index for r in results])
    node_list = torch.cat([r.output.node for r in results])
    row_list = torch.cat([r.output.row for r in results])
    col_list = torch.cat([r.output.col for r in results])
    eids_list = torch.cat([r.output.edge for r in results]) if self.with_edge else None

    return SamplerOutput(node_list, row_list, col_list, eids_list)

  async def _sample_one_hop(
    self,
    srcs: torch.Tensor,
    num_nbr: int,
    etype: Optional[EdgeType]
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
    device = torch.device(type='cpu') #self.device
    ##device = self.device
    srcs = srcs.to(device)
    orders = torch.arange(srcs.size(0), dtype=torch.long, device=device)
    src_ntype = etype[0] if etype is not None else None
    
    print(f"\n\n\n\n XXXXXXXXXXXXXXXX-------- DistNSampler: async _sample_one_hop(), device={device} , srcs.size(0)={srcs.size(0)}, srcs={srcs}, src_ntype={src_ntype}, num_nbr={num_nbr} ------")
    partition_ids = self.dist_graph.get_partition_ids_from_nids(srcs, src_ntype)
    partition_ids = partition_ids.to(device)

    partition_results: List[PartialNeighborOutput] = []
    remote_orders_list: List[torch.Tensor] = []
    futs: List[torch.futures.Future] = []

    print(f"-------- DistNSampler: async _sample_one_hop() enter multi-partition sample_one_hop() -------")

    for i in range(self.dist_graph.num_partitions):
    #for i in range(self.data[1]): #.num_partitions):
      pidx = (
        #(self.data[2] + i) % self.data[1]
        (self.dist_graph.partition_idx + i) % self.dist_graph.num_partitions
      )
      p_mask = (partition_ids == pidx)
      p_ids = torch.masked_select(srcs, p_mask)


      #*print(f"-------- DistNSampler: async _sample_one_hop(), partition i={i}, pidx={pidx}, p_ids.shape={p_ids.shape} -----")
      print(f"-------- DistNSampler: async _sample_one_hop(), partition i={i}, pidx={pidx}, p_ids={p_ids}, p_ids.shape={p_ids.shape} -----")
      if p_ids.shape[0] > 0:
        p_orders = torch.masked_select(orders, p_mask)
        if pidx == self.dist_graph.num_partitions:
          
          print(f"----000---- DistNSampler: async _sample_one_hop(), pidx={pidx} ------")
          p_nbr_out = self._sampler.sample_one_hop(p_ids, num_nbr, etype)
          print(f"----000.2---- DistNSampler: async _sample_one_hop(), p_nbr_out={p_nbr_out} ------")
          partition_results.append(PartialNeighborOutput(p_orders, p_nbr_out))
        else:
          print(f"----111---- DistNSampler: async _sample_one_hop(), pidx={pidx}, p_orders.shape={p_orders.shape} ------")
          remote_orders_list.append(p_orders)
          to_worker = self.rpc_router.get_to_worker(pidx)
          print(f"----111.1---- DistNSampler: async _sample_one_hop(), to_worker={to_worker} ------")
          temp_futs = rpc_async(to_worker,
                                        self.rpc_sample_callee_id,
                                        args=(p_ids.cpu(), num_nbr, etype))
          print(f"----111.2---- DistNSampler: async _sample_one_hop(), temp_futs={temp_futs.wait()} ------")
          futs.append(temp_futs)
    
    print(f"-----333--- DistNSampler: async _sample_one_hop() without remote sampling results -------")
    # Without remote sampling results.
    if len(remote_orders_list) == 0:
      return partition_results[0].output
    # With remote sampling results.
    res_fut_list = await wrap_torch_future(torch.futures.collect_all(futs))
    for i, res_fut in enumerate(res_fut_list):
      print(f"-------- DistNSampler: async _sample_one_hop() res_fut={res_fut.wait()} -------")
      
      partition_results.append(
        PartialNeighborOutput(
          index=remote_orders_list[i],
          output=res_fut.wait()
          #output=res_fut.wait().to(device)
        )
      )
    
    print(f"-------- DistNSampler: async _sample_one_hop() before stitching -----------------  partition_results={partition_results}-------")
    print("\n\n\n\n")
    return self._stitch_sample_results(srcs, partition_results)

  async def _colloate_fn(
    self,
    output: Union[SamplerOutput, HeteroSamplerOutput]
  ) -> SamplerOutput:
    r""" Collect labels and features for the sampled subgrarph if necessary,
    and put them into a sample message.
    """
    #is_hetero = (self.dist_graph.data_cls == 'hetero')
    result_map = {}
    # if isinstance(output.metadata, dict):
      #scan kv and add metadata
      # input_type = output.metadata.get('input_type', '')
      # batch_size = output.metadata.get('bs', 1)
      # result_map['meta'] = torch.LongTensor([int(is_hetero), batch_size])
      # # output.metadata.pop('input_type', '')
      # # output.metadata.pop('bs', 1)
      # for k, v in output.metadata.items():
      #   result_map[k] = v
  
    #if is_hetero:
    if(self.dist_graph.meta["is_hetero"]):
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
        #fut = self.dist_node_feature.async_get(output.node)
        
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
        #fut = self.dist_edge_feature.async_get(eids)

        fut = self.dist_edge_feature.lookup_features(is_node_feat=False, ids=eids)
        efeats = await wrap_torch_future(fut)
        output.metadata['efeats'] = efeats
      else:
        output.metadata['efeats'] = None
      
      #print(f"------- 777.4 ----- DistNSampler: _colloate_fn()  return -------")

    return output #result_map
  


# Sampling Utilities ##########################################################

def sample(
    inputs: NodeSamplerInput,
    sample_fn: Callable,
    _sample_fn: Callable,
    ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
  #print(f"------  Sampler PID-{mp.current_process().pid}:  sample_from_nodes -------")
  return sample_fn(inputs, _sample_fn)


def close_sampler(worker_id, sampler):
  print(f"Closing rpc in {repr(sampler)} worker-id {worker_id}")
  try:
    sampler.event_loop.shutdown_loop()
  except AttributeError:
    pass
  shutdown_rpc(graceful=True)
