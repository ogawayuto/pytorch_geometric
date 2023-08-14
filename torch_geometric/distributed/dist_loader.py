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
import atexit
import torch
import torch.multiprocessing as mp
import os
from typing import Callable, Optional, Tuple, Dict, Union, List

from torch_geometric.data import Data, HeteroData
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput

from .rpc import init_rpc, global_barrier
from .dist_mixin import RPCMixin
from .dist_neighbor_sampler import DistNeighborSampler, close_sampler
from .dist_context import DistContext, DistRole
#from ..channel import ChannelBase
from torch_geometric.typing import EdgeType, InputNodes, OptTensor, as_str


class DistLoader():  # , RPCMixin):
    r"""
    #TODO: Update readme
    # A distributed loader that preform sampling from nodes.

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
                 neighbor_sampler: DistNeighborSampler,
                 current_ctx: DistContext,
                 rpc_worker_names: Dict[DistRole, List[str]],
                 master_addr: str,
                 master_port: Union[int, str],
                 channel: mp.Queue(), #Optional[Union[ChannelBase, mp.Queue()]],
                 num_rpc_threads: Optional[int] = 16,
                 rpc_timeout: Optional[int] = 180,
                 device: Optional[torch.device] = torch.device('cpu'),
                 **kwargs,
                 ):

        self.neighbor_sampler = neighbor_sampler
        self.channel = channel
        self.current_ctx = current_ctx
        self.rpc_worker_names = rpc_worker_names
        if master_addr is not None:
            self.master_addr = str(master_addr)
        elif os.environ.get('MASTER_ADDR') is not None:
            self.master_addr = os.environ['MASTER_ADDR']
        else:
            raise ValueError(
                f"'{self.__class__.__name__}': missing master address "
                "for rpc communication, try to provide it or set it "
                "with environment variable 'MASTER_ADDR'")
        if master_port is not None:
            self.master_port = int(master_port)
        elif os.environ.get('MASTER_PORT') is not None:
            self.master_port = int(os.environ['MASTER_PORT']) + 1
        else:
            raise ValueError(
                f"'{self.__class__.__name__}': missing master port "
                "for rpc communication, try to provide it or set it "
                "with environment variable 'MASTER_ADDR'")

        self.num_rpc_threads = num_rpc_threads
        if self.num_rpc_threads is not None:
            assert self.num_rpc_threads > 0
        self.rpc_timeout = rpc_timeout
        if self.rpc_timeout is not None:
            assert self.rpc_timeout > 0

        self.device = device
        self.batch_size = kwargs.get('batch_size', 64)
        self.num_workers = kwargs.get('num_workers', 0)

        self.pid = mp.current_process().pid

        if self.num_workers == 0:
            self.init_fn(0)

    def init_fn(self, worker_id):
        try:
            print(f">>> EXECUTING init_fn() in _worker_loop() of {repr(self.neighbor_sampler)} worker_id-{worker_id}: ")
            num_sampler_proc = (self.num_workers if self.num_workers > 0 else 1)
            self.current_ctx_worker = DistContext(
                world_size=self.current_ctx.world_size * num_sampler_proc,
                rank=self.current_ctx.rank * num_sampler_proc + worker_id,
                global_world_size=self.current_ctx.world_size * num_sampler_proc,
                global_rank=self.current_ctx.rank * num_sampler_proc + worker_id,
                group_name='mp_sampling_worker')

            print(f"DONE: set DistContext() {self.current_ctx_worker}")

            self.sampler_rpc_worker_names = {}
            init_rpc(
                current_ctx=self.current_ctx_worker,
                rpc_worker_names=self.sampler_rpc_worker_names,
                master_addr=self.master_addr,
                master_port=self.master_port,
                num_rpc_threads=self.num_rpc_threads,
                rpc_timeout=self.rpc_timeout
            )
            print(f"DONE: init_rpc()")
            self.neighbor_sampler.register_sampler_rpc()
            print(f"DONE: register_sampler_rpc()")
            self.neighbor_sampler.init_event_loop()
            print(f"DONE: init_event_loop()")
            # close rpc & worker group at exit
            atexit.register(close_sampler, worker_id, self.neighbor_sampler)
            # wait for all workers to init
            global_barrier()
            print(f">>> FINISHED EXECUTING init_fn()")

        except RuntimeError:
            raise RuntimeError(
                f"init_fn() defined in {repr(self)} didn't initialize the worker_loop of {repr(self.neighbor_sampler)}")

    def channel_get(self) -> Union[Data, HeteroData]:
        if self.channel and not self.filter_per_worker:
            out = self.channel.get()
            #print(f'{repr(self)} retrieved Sampler result from PyG MSG channel')
        return out

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

            #print(f"----9999 dist_loader: filter_fn ----- out={out} ")
            edge_index = torch.stack([out.row, out.col])
            data = Data(x=out.metadata['nfeats'],
                        edge_index=edge_index,
                        edge_attr=out.metadata['efeats'],
                        y=out.metadata['nlabels']
                        )
            data.edge = out.edge
            data.node = out.node
            data.batch = out.batch
            #data.batch_size = out.batch.numel() if data.batch is not None else 0
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()-PID{self.pid}@{self.device}"
