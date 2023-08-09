# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Optional

import torch

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
from torch_geometric.data import Data, HeteroData
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import EdgeType, InputNodes, OptTensor, as_str

class DistLinkNeighborLoader(LinkLoader, DistLoader):
    # TODO: Update readme
    r""" A distributed loader that preform sampling from edges.

    Args:
      data (DistDataset, optional): The ``DistDataset`` object of a partition of
        graph data and feature data, along with distributed patition books. The
        input dataset must be provided in non-server distribution mode.
      num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]):
        The number of neighbors to sample for each node in each iteration.
        In heterogeneous graphs, may also take in a dictionary denoting
        the amount of neighbors to sample for each individual edge type.
      batch_size (int): How many samples per batch to load (default: ``1``).
      edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
        The edge indices, holding source and destination nodes to start
        sampling from.
        If set to :obj:`None`, all edges will be considered.
        In heterogeneous graphs, needs to be passed as a tuple that holds
        the edge type and corresponding edge indices.
        (default: :obj:`None`)
      edge_label (Tensor, optional): The labels of edge indices from which to
        start sampling from. Must be the same length as
        the :obj:`edge_label_index`. (default: :obj:`None`)
      neg_sampling (NegativeSampling, optional): The negative sampling
        configuration.
        For negative sampling mode :obj:`"binary"`, samples can be accessed
        via the attributes :obj:`edge_label_index` and :obj:`edge_label` in
        the respective edge type of the returned mini-batch.
        In case :obj:`edge_label` does not exist, it will be automatically
        created and represents a binary classification task (:obj:`0` =
        negative edge, :obj:`1` = positive edge).
        In case :obj:`edge_label` does exist, it has to be a categorical
        label from :obj:`0` to :obj:`num_classes - 1`.
        After negative sampling, label :obj:`0` represents negative edges,
        and labels :obj:`1` to :obj:`num_classes` represent the labels of
        positive edges.
        Note that returned labels are of type :obj:`torch.float` for binary
        classification (to facilitate the ease-of-use of
        :meth:`F.binary_cross_entropy`) and of type
        :obj:`torch.long` for multi-class classification (to facilitate the
        ease-of-use of :meth:`F.cross_entropy`).
        For negative sampling mode :obj:`"triplet"`, samples can be
        accessed via the attributes :obj:`src_index`, :obj:`dst_pos_index`
        and :obj:`dst_neg_index` in the respective node types of the
        returned mini-batch.
        :obj:`edge_label` needs to be :obj:`None` for :obj:`"triplet"`
        negative sampling mode.
        If set to :obj:`None`, no negative sampling strategy is applied.
        (default: :obj:`None`)
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
                 with_edge: bool = True,
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
                 concurrency: int = 4,
                 collect_features: bool = True,
                 async_sampling: bool = True,
                 device: Optional[torch.device] = torch.device('cpu'),
                 **kwargs
                 ):

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
                data=data,  # data.graph?
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
        LinkLoader.__init__(self,
                            # Tuple[FeatureStore, GraphStore]
                            data=data,
                            link_sampler=neighbor_sampler,
                            edge_label_index=edge_label_index,
                            edge_label=edge_label,
                            neg_sampling=neg_sampling,
                            neg_sampling_ratio=neg_sampling_ratio,
                            transform=transform,
                            transform_sampler_output=transform_sampler_output,
                            filter_per_worker=filter_per_worker,
                            custom_init=self.init_fn,
                            **kwargs
                            )
        
    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        r"""Joins the sampled nodes with their corresponding features,
        returning the resulting :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object to be used downstream.
        """
        if self.channel and not self.filter_per_worker:
            out = self.channel.get()
            print(f'{repr(self)} retrieved Sampler result from PyG MSG channel')

        if isinstance(out, SamplerOutput):
            edge_index = torch.stack([out.row, out.col])
            data = Data(x=out.metadata['nfeats'],
                        edge_index=edge_index,
                        edge_attr=out.metadata['efeats'],
                        y=out.metadata['nlabels']
                        )
            data.edge = out.edge
            data.node = out.node
            data.batch = out.batch
            data.batch_size = out.metadata['bs']

            if 'edge_label_index' in out.metadata:
                # binary negative sampling
                # In this case, we reverse the edge_label_index and put it into the
                # reversed edgetype subgraph
                edge_label_index = torch.stack(
                    (out.metadata['edge_label_index'][1],
                     out.metadata['edge_label_index'][0]), dim=0)
                data.edge_label_index = edge_label_index
                data.edge_label = out.metadata['edge_label']
            elif 'src_index' in out.metadata:
                # triplet negative sampling
                # In this case, src_index and dst_pos/neg_index fields follow the nodetype
                data.src_index = out.metadata['src_index']
                data.dst_pos_index = out.metadata['dst_pos_index']
                data.dst_neg_index = out.metadata['dst_neg_index']
            else:
                pass

        elif isinstance(out, HeteroSamplerOutput):
            # TODO: Refactor hetero
          #   def to_hetero_data(
          #   hetero_sampler_out: HeteroSamplerOutput,
          #   batch_label_dict: Optional[Dict[NodeType, torch.Tensor]] = None,
          #   node_feat_dict: Optional[Dict[NodeType, torch.Tensor]] = None,
          #   edge_feat_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
          #   **kwargs
          # ) -> HeteroData:
            node_dict, row_dict, col_dict, edge_dict = {}, {}, {}, {}
            nfeat_dict, efeat_dict = {}, {}

            for ntype in self._node_types:
                ids_key = f'{as_str(ntype)}.ids'
                if ids_key in out:
                    node_dict[ntype] = out[ids_key].to(self.to_device)
                    nfeat_key = f'{as_str(ntype)}.nfeats'
                if nfeat_key in out:
                    nfeat_dict[ntype] = out[nfeat_key].to(self.to_device)

            for etype_str, rev_etype in self._etype_str_to_rev.items():
                rows_key = f'{etype_str}.rows'
                cols_key = f'{etype_str}.cols'
                if rows_key in out:
                    # The edge index should be reversed.
                    row_dict[rev_etype] = out[cols_key].to(self.to_device)
                    col_dict[rev_etype] = out[rows_key].to(self.to_device)
                    eids_key = f'{etype_str}.eids'
                if eids_key in out:
                    edge_dict[rev_etype] = out[eids_key].to(self.to_device)
                    efeat_key = f'{etype_str}.efeats'
                if efeat_key in out:
                    efeat_dict[rev_etype] = out[efeat_key].to(self.to_device)

            batch_dict = {
                self.input_type: node_dict[self.input_type]
                [: self.batch_size]}
            output = HeteroSamplerOutput(
                node_dict, row_dict, col_dict, edge_dict
                if len(edge_dict) else None, batch_dict,
                edge_types=self._edge_types, device=self.to_device)

            if len(nfeat_dict) == 0:
                nfeat_dict = None
            if len(efeat_dict) == 0:
                efeat_dict = None

            batch_labels_key = f'{self.input_type}.nlabels'
            if batch_labels_key in out:
                batch_labels = out[batch_labels_key].to(self.to_device)
            else:
                batch_labels = None
            label_dict = {self.input_type: batch_labels}

            # res_data = to_hetero_data(
            #     output, label_dict, nfeat_dict, efeat_dict)

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(out)}'")
        return data
