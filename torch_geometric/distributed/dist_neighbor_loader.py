import torch
from typing import Callable, Optional, Tuple, Dict, Union, List

from torch_geometric.data import Data, HeteroData
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import EdgeType, InputNodes, OptTensor, as_str
from torch_geometric.loader.node_loader import NodeLoader

from .dist_loader import DistLoader
from .dist_neighbor_sampler import DistNeighborSampler
from .local_graph_store import LocalGraphStore
from .local_feature_store import LocalFeatureStore
from .dist_context import DistContext, DistRole

from torch_geometric.sampler.base import SubgraphType


class DistNeighborLoader(NodeLoader, DistLoader):
    r""" A distributed loader that preform sampling from nodes.
    Args:
      data (DistDataset, optional): The ``DistDataset`` object of a partition of
        graph data and feature data, along with distributed patition books. The
        input dataset must be provided in non-server distribution mode.
      num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]):
        The number of neighbors to sample for each node in each iteration.
        In heterogeneous graphs, may also take in a dictionary denoting
        the amount of neighbors to sample for each individual edge type.
      input_nodes (torch.Tensor or Tuple[str, torch.Tensor]): The node seeds for
        which neighbors are sampled to create mini-batches. In heterogeneous
        graphs, needs to be passed as a tuple that holds the node type and
        node seeds.
      batch_size (int): How many samples per batch to load (default: ``1``).
      shuffle (bool): Set to ``True`` to have the data reshuffled at every
        epoch (default: ``False``).
      drop_last (bool): Set to ``True`` to drop the last incomplete batch, if
        the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last
        batch will be smaller. (default: ``False``).
      with_edge (bool): Set to ``True`` to sample with edge ids and also include
        them in the sampled results. (default: ``False``).
      collect_features (bool): Set to ``True`` to collect features for nodes
        of each sampled subgraph. (default: ``False``).
      to_device (torch.device, optional): The target device that the sampled
        results should be copied to. If set to ``None``, the current cuda device
        (got by ``torch.cuda.current_device``) will be used if available,
        otherwise, the cpu device will be used. (default: ``None``).
      worker_options (optional): The options for launching sampling workers.
        (1) If set to ``None`` or provided with a ``CollocatedDistWorkerOptions``
        object, a single collocated sampler will be launched on the current
        process, while the separate sampling mode will be disabled . (2) If
        provided with a ``MpDistWorkerOptions`` object, the sampling workers will
        be launched on spawned subprocesses, and a share-memory based channel
        will be created for sample message passing from multiprocessing workers
        to the current loader. (3) If provided with a ``RemoteDistWorkerOptions``
        object, the sampling workers will be launched on remote sampling server
        nodes, and a remote channel will be created for cross-machine message
        passing. (default: ``None``).
      """

    def __init__(self,
                 data: Tuple[Dict, int, int, LocalGraphStore, LocalFeatureStore, torch.Tensor, torch.Tensor],
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
                 transform_sampler_output: Optional[Callable] = None,
                 is_sorted: bool = False,
                 directed: bool = True,  # Deprecated.
                 with_edge: bool = False,
                 concurrency: int = 0,
                 collect_features: bool = True,
                 filter_per_worker: Optional[bool] = None,
                 async_sampling: bool = True,
                 device: torch.device = torch.device('cpu'),
                 **kwargs,
                 ):

        if input_time is not None and time_attr is None:
            raise ValueError("Received conflicting 'input_time' and "
                             "'time_attr' arguments: 'input_time' is set "
                             "while 'time_attr' is not set.")

        channel = torch.multiprocessing.Queue() if async_sampling else None

        if neighbor_sampler is None:
            neighbor_sampler = DistNeighborSampler(
                data=data,  # data.graph?
                current_ctx=current_ctx,
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
                collect_features=collect_features
            )

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
                            # Tuple[FeatureStore, GraphStore]
                            data=data,
                            node_sampler=neighbor_sampler,
                            input_nodes=input_nodes,
                            input_time=input_time,
                            transform=transform,
                            transform_sampler_output=transform_sampler_output,
                            filter_per_worker=filter_per_worker,
                            custom_init=self.init_fn,
                            custom_filter=self.filter_fn,
                            **kwargs
                            )
        
    #TODO: move filter_fn here    
    # def filter_fn():
    #   pass
