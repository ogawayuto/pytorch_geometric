from typing import Optional

import torch
import logging 

from torch_geometric.sampler.base import (
    EdgeSamplerInput, SamplingType, SamplingConfig, NegativeSampling
)
from torch_geometric.typing import InputEdges, NumNeighbors
from typing import Callable, Optional, Tuple, Dict, Union, List
from .local_graph_store import LocalGraphStore
from .local_feature_store import LocalFeatureStore
from .dist_loader import DistLoader
from .dist_neighbor_sampler import DistNeighborSampler
from torch_geometric.sampler.base import SubgraphType
from .dist_context import DistContext, DistRole
from torch_geometric.loader.link_loader import LinkLoader
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from ..typing import Tuple, Dict, Union
from torch_geometric.data import Data, HeteroData, GraphStore, FeatureStore
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import EdgeType, InputNodes, OptTensor, as_str
from torch_geometric.loader.utils import filter_custom_store

class DistLinkNeighborLoader(LinkLoader, DistLoader):
    r""" A distributed loader that preform sampling from edges.
    """

    def __init__(self,
                 data: Tuple[LocalFeatureStore, LocalGraphStore],
                 num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
                 master_addr: str,
                 master_port: Union[int, str],
                 current_ctx: DistContext,
                 rpc_worker_names: Dict[DistRole, List[str]],
                 neighbor_sampler: Optional[DistNeighborSampler] = None,
                 edge_label_index: InputEdges = None,
                 edge_label: OptTensor = None,
                 edge_label_time: OptTensor = None,
                 replace: bool = False,
                 subgraph_type: Union[SubgraphType, str] = 'directional',
                 disjoint: bool = False,
                 temporal_strategy: str = 'uniform',
                 neg_sampling: Optional[NegativeSampling] = None,
                 neg_sampling_ratio: Optional[Union[int, float]] = None,
                 time_attr: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 transform_sampler_output: Optional[Callable] = None,
                 is_sorted: bool = False,
                 filter_per_worker: Optional[bool] = None,
                 directed: bool = True,  # Deprecated.
                 concurrency: int = 1,
                 async_sampling: bool = True,
                 device: Optional[torch.device] = torch.device('cpu'),
                 **kwargs
                 ):

        
        assert (isinstance(data[0], LocalFeatureStore) and (
            data[1], LocalGraphStore)), "Data needs to be Tuple[LocalFeatureStore, LocalGraphStore]"
        
        assert concurrency >= 1, "RPC concurrency must be greater than 1."
        
        channel = torch.multiprocessing.Queue() if async_sampling else None

        if (edge_label_time is not None) != (time_attr is not None):
            raise ValueError(
                f"Received conflicting 'edge_label_time' and 'time_attr' "
                f"arguments: 'edge_label_time' is "
                f"{'set' if edge_label_time is not None else 'not set'} "
                f"while 'time_attr' is "
                f"{'set' if time_attr is not None else 'not set'}. "
                f"Both arguments must be provided for temporal sampling."
            )

        if neighbor_sampler is None:
            neighbor_sampler = DistNeighborSampler(
                data=data,
                current_ctx=current_ctx,
                rpc_worker_names=rpc_worker_names,
                num_neighbors=num_neighbors,
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
            )
            
        self.neighbor_sampler = neighbor_sampler

        DistLoader.__init__(self,
                            channel=channel,
                            master_addr=master_addr,
                            master_port=master_port,
                            current_ctx=current_ctx,
                            rpc_worker_names=rpc_worker_names,
                            **kwargs
                            )
        LinkLoader.__init__(self,
                            data=data,
                            link_sampler=neighbor_sampler,
                            edge_label_index=edge_label_index,
                            edge_label=edge_label,
                            neg_sampling=neg_sampling,
                            neg_sampling_ratio=neg_sampling_ratio,
                            transform=transform,
                            transform_sampler_output=transform_sampler_output,
                            filter_per_worker=filter_per_worker,
                            worker_init_fn=self.worker_init_fn,
                            transform_sampler_output=self.channel_get,
                            **kwargs
                            )
      
    def __repr__(self):
      return DistLoader.__repr__(self)