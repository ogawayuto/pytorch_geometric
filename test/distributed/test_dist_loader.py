import os.path as osp
import subprocess
from time import sleep
import socket
from typing import Dict, List

import numpy as np
import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.distributed.dist_loader import DistLoader
from torch_geometric.distributed.dist_neighbor_loader import DistNeighborLoader
from torch_geometric.distributed.dist_link_neighbor_loader import DistLinkNeighborLoader

from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.distributed import LocalGraphStore

from torch_geometric.testing import (
    MyFeatureStore,
    MyGraphStore,
    get_random_edge_index,
    onlyLinux,
    onlyNeighborSampler,
    onlyOnline,
    withPackage,
)
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
from torch_geometric.utils import (
    is_undirected,
    sort_edge_index,
    to_torch_csr_tensor,
    to_undirected,
)


def is_subset(subedge_index, edge_index, src_idx, dst_idx):
    num_nodes = int(edge_index.max()) + 1
    idx = num_nodes * edge_index[0] + edge_index[1]
    subidx = num_nodes * src_idx[subedge_index[0]] + dst_idx[subedge_index[1]]
    mask = torch.from_numpy(np.isin(subidx.cpu().numpy(), idx.cpu().numpy()))
    return int(mask.sum()) == mask.numel()

def test_homo(
    world_size: int,
    rank: int,
    master_addr: str,
    master_port: int,
    device = torch.device('cpu'),
    rpc_worker_names: Dict[DistRole, List[str]] = {}
    ):
    
    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-loader-homo-test'
        )


    torch.manual_seed(12345)

    node_id = torch.randperm(100)
    x = torch.randn(100, 32)
    y = torch.randint(0, 2, (100, ))
    edge_id = torch.randperm(200)
    edge_attr = torch.randn(200, 16)
    edge_index = get_random_edge_index(100, 100, 200)
    feat_store = LocalFeatureStore.from_data(node_id, x, y, edge_id, edge_attr)
    graph_store = LocalGraphStore.from_data(edge_id, edge_index, num_nodes=100)
    graph_store.meta = {"is_hetero" : False}
    feat_store.meta = graph_store.meta
    input_nodes = torch.range(0,99, dtype = torch.int32)
    data = (graph_store, feat_store)
    
    loader = DistNeighborLoader(
        data,
        num_neighbors=[5] * 2,
        batch_size=20,
        input_nodes=input_nodes,
        master_addr=master_addr,
        master_port=master_port,
        current_ctx=current_ctx,
        rpc_worker_names=rpc_worker_names,
        device=device
    )

    assert str(loader) == 'DistNeighborLoader()'
    assert len(loader) == 5

    # batch = loader([0])
    # assert isinstance(batch, Data)
    # assert batch.n_id[:1].tolist() == [0]

    for i, batch in enumerate(loader):
        assert isinstance(batch, Data)
        assert batch.x.device == device
        assert batch.x.size(0) <= 100
        assert batch.n_id.size() == (batch.num_nodes, )
        assert batch.input_id.numel() == batch.batch_size == 20
        assert batch.x.min() >= 0 and batch.x.max() < 100
        assert batch.edge_index.device == device
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        assert batch.edge_attr.device == device
        assert batch.edge_attr.size(0) == batch.edge_index.size(1)

        # Input nodes are always sampled first:
        assert torch.equal(
            batch.x[:batch.batch_size],
            torch.arange(i * batch.batch_size, (i + 1) * batch.batch_size,
                         device=device),
        )
        
@onlyLinux
def test_dist_neighbor_loader():

            
    mp_context = torch.multiprocessing.get_context('spawn')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    addr='localhost'


    w0 = mp_context.Process(target=test_homo,
                            args=(2, 0, addr, port))
    w1 = mp_context.Process(target=test_homo,
                            args=(2, 1, addr, port))

    w0.start()
    w1.start()
    w0.join()
    w1.join()