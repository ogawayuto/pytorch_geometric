import torch
from typing import Callable, Optional, Tuple, Dict, Union, List

from torch_geometric.typing import EdgeType, InputNodes, OptTensor
from torch_geometric.distributed.dist_neighbor_sampler import DistNeighborSampler
from torch_geometric.distributed.dist_loader import DistLoader
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from torch_geometric.distributed.dist_context import DistRole, DistContext
from torch_geometric.loader.node_loader import NodeLoader
from torch_geometric.sampler.base import SubgraphType


class DistNeighborLoader(NodeLoader, DistLoader):
    r""" A distributed loader that preform sampling from nodes.
    Args:

      """

    def __init__(self,
                 data: Tuple[LocalFeatureStore, LocalGraphStore],
                 num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
                 master_addr: str,
                 master_port: Union[int, str],
                 current_ctx: DistContext,
                 rpc_worker_names: Dict[DistRole, List[str]],
                 neighbor_sampler: Optional[DistNeighborSampler] = None,
                 input_nodes: InputNodes = None,
                 input_time: OptTensor = None,
                 replace: bool = False,
                 subgraph_type: Union[SubgraphType, str] = 'directional',
                 disjoint: bool = False,
                 temporal_strategy: str = 'uniform',
                 time_attr: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 is_sorted: bool = False,
                 directed: bool = True,  # Deprecated.
                 with_edge: bool = True,
                 concurrency: int = 1,
                 collect_features: bool = True,
                 filter_per_worker: Optional[bool] = False,
                 async_sampling: bool = True,
                 device: torch.device = torch.device('cpu'),
                 **kwargs,
                 ):

        assert (isinstance(data[0], LocalFeatureStore) and (
            data[1], LocalGraphStore)), "Data needs to be Tuple[LocalFeatureStore, LocalGraphStore]"
        
        assert concurrency >= 1, "concurrency must be greater than 1"
        
        if input_time is not None and time_attr is None:
            raise ValueError("Received conflicting 'input_time' and "
                             "'time_attr' arguments: 'input_time' is set "
                             "while 'time_attr' is not set.")
        
        channel = torch.multiprocessing.Queue() if async_sampling else None

        if neighbor_sampler is None:
            neighbor_sampler = DistNeighborSampler(
                data=data,
                current_ctx=current_ctx,
                rpc_worker_names=rpc_worker_names,
                num_neighbors=num_neighbors,
                with_edge=with_edge,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
                directed=directed,
                device=device,
                channel=channel,
                concurrency=concurrency,
                collect_features=collect_features,
            )
        self.neighbor_sampler = neighbor_sampler
                     
        DistLoader.__init__(self,
                            neighbor_sampler=neighbor_sampler,
                            channel=channel,
                            master_addr=master_addr,
                            master_port=master_port,
                            current_ctx=current_ctx,
                            rpc_worker_names=rpc_worker_names,
                            **kwargs
                            )
        NodeLoader.__init__(self,
                            data=data,
                            node_sampler=neighbor_sampler,
                            input_nodes=input_nodes,
                            input_time=input_time,
                            transform=transform,
                            transform_sampler_output=self.channel_get,
                            filter_per_worker=filter_per_worker,
                            worker_init_fn=self.worker_init_fn,
                            **kwargs
                            )

    def __repr__(self):
        return DistLoader.__repr__(self)
