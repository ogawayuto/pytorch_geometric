import socket
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.distributed.partition import load_partition_info

from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.distributed import LocalGraphStore
from torch_geometric.distributed.dist_neighbor_sampler import DistNeighborSampler
from torch_geometric.distributed.dist_neighbor_loader import DistNeighborLoader

from torch_geometric.testing import (
    get_random_edge_index,
    onlyLinux,
    withPackage,
)
from torch_geometric.typing import WITH_METIS


def is_subset(subedge_index, edge_index, src_idx, dst_idx):
    num_nodes = int(edge_index.max()) + 1
    idx = num_nodes * edge_index[0] + edge_index[1]
    subidx = num_nodes * src_idx[subedge_index[0]] + dst_idx[subedge_index[1]]
    mask = torch.from_numpy(np.isin(subidx.cpu().numpy(), idx.cpu().numpy()))
    return int(mask.sum()) == mask.numel()

@pytest.mark.parametrize('num_workers', [0,2], 'concurrency', [0,1,2])
def homo_dist_neighbor_loader(
    tmp_path: str,
    world_size: int,
    rank: int,
    master_addr: str,
    master_port: int,
    num_workers: int,
    concurrency: int,
    ):
    device = torch.device('cpu')    
    graph_store = LocalGraphStore.from_partition(tmp_path, pid=rank)
    feat_store = LocalFeatureStore.from_partition(tmp_path, pid=rank)
    (
        meta, num_partitions, partition_idx, node_pb, edge_pb
    ) = load_partition_info(tmp_path, rank)
    graph_store.partition_idx = partition_idx
    graph_store.num_partitions = num_partitions
    graph_store.node_pb = node_pb
    graph_store.edge_pb = edge_pb
    graph_store.meta = meta
    edge_attrs = graph_store.get_all_edge_attrs()[0]
    graph_store.labels = torch.arange(edge_attrs.size[0])
    
    feat_store.partition_idx = partition_idx
    feat_store.num_partitions = num_partitions
    feat_store.feature_pb = node_pb 
    feat_store.meta = meta
    
    data = (feat_store, graph_store)
    input_nodes = feat_store.get_global_id(None)
    
    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-loader-homo-test'
        )
    
    loader = DistNeighborLoader(
        data,
        num_neighbors=[5],
        batch_size=10,
        num_workers=num_workers,
        input_nodes=input_nodes,
        master_addr=master_addr,
        master_port=master_port,
        current_ctx=current_ctx,
        rpc_worker_names = {},
        concurrency=concurrency,
        collect_features=True,
        device=device,
        drop_last = False,
        async_sampling = False
    )

    assert 'DistNeighborLoader()' in str(loader)
    for value in loader.sampler_rpc_worker_names.values():
        if loader.num_workers == 0:
            assert len(value) == 2
        else:
            assert len(value) == 2 * loader.num_workers
        
    assert isinstance(loader.neighbor_sampler, DistNeighborSampler)
        
    for i, batch in enumerate(loader):
        assert isinstance(batch, Data)
        assert batch.x.device == device
        assert batch.x.size(0) <= 100
        #assert batch.n_id.size() == (batch.num_nodes, )
        #assert batch.input_id.numel() == batch.batch_size == 10
        #assert batch.x.min() >= 0 and batch.x.max() < 100
        assert batch.edge_index.device == device
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        #assert batch.edge_attr.device == device
        #assert batch.edge_attr.size(0) == batch.edge_index.size(1)

        # Input nodes are always sampled first:
        # assert torch.equal(
        #     batch.x[:batch.batch_size],
        #     torch.arange(i * batch.batch_size, (i + 1) * batch.batch_size,
        #                  device=device),
        # )
        
@onlyLinux
@pytest.mark.skipif(not WITH_METIS, reason='Not compiled with METIS support')
def test_dist_neighbor_loader(tmp_path):
       
    mp_context = torch.multiprocessing.get_context('spawn')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    addr='localhost'
    
    data = FakeDataset(
        num_graphs=1,
        avg_num_nodes=100,
        avg_degree=5,
        edge_dim=1)[0]
    world_size = 2
    partitioner = Partitioner(data, world_size, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(target=homo_dist_neighbor_loader,
                            args=(tmp_path, world_size, 0, addr, port))
    
    w1 = mp_context.Process(target=homo_dist_neighbor_loader,
                            args=(tmp_path, world_size, 1, addr, port))

    w0.start()
    w1.start()
    w0.join()
    w1.join()
    
    
