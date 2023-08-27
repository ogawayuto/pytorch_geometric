import socket
import pytest
import torch
import torch.multiprocessing as mp

from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.distributed.partition import load_partition_info

from torch_geometric.distributed.dist_context import DistContext
from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.distributed import LocalGraphStore
from torch_geometric.distributed.dist_neighbor_sampler import DistNeighborSampler
from torch_geometric.distributed.dist_neighbor_loader import DistNeighborLoader
from torch_geometric.distributed.dist_link_neighbor_loader import DistLinkNeighborLoader
from torch_geometric.sampler import NegativeSampling

from torch_geometric.testing import (
    get_random_edge_index,
    onlyLinux,
    withPackage,
)
from torch_geometric.typing import WITH_METIS


def create_dist_data(tmp_path, rank):
    graph_store = LocalGraphStore.from_partition(tmp_path, pid=rank)
    feat_store = LocalFeatureStore.from_partition(tmp_path, pid=rank)
    (
        meta, num_partitions, partition_idx, node_pb, edge_pb
    ) = load_partition_info(tmp_path, rank)
    if meta['is_hetero']:
        node_pb = torch.cat(list(node_pb.values()))
        edge_pb = torch.cat(list(edge_pb.values()))
    else:
        edge_attrs = graph_store.get_all_edge_attrs()[0]
        graph_store.labels = torch.arange(edge_attrs.size[0])

    graph_store.partition_idx = partition_idx
    graph_store.num_partitions = num_partitions
    graph_store.node_pb = node_pb
    graph_store.edge_pb = edge_pb
    graph_store.meta = meta

    feat_store.partition_idx = partition_idx
    feat_store.num_partitions = num_partitions
    feat_store.node_feat_pb = node_pb
    feat_store.edge_feat_pb = edge_pb
    feat_store.meta = meta

    data = (feat_store, graph_store)
    if meta['is_hetero']:
        input_nodes = ('v0', feat_store.get_global_id('v0'))
    else:
        input_nodes = feat_store.get_global_id(None)
    return data, input_nodes


def dist_link_neighbor_loader_homo(
    tmp_path: str,
    world_size: int,
    rank: int,
    master_addr: str,
    master_port: int,
    num_workers: int,
    concurrency: int,
    async_sampling: bool,
    device=torch.device('cpu')
):

    data, _ = create_dist_data(tmp_path, rank)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-loader-test'
    )
    edge_label_index_train = (
        None, torch.stack(
            [data[1].get_edge_index((None, 'coo'))[0],
             data[1].get_edge_index((None, 'coo'))[1]],
            dim=0))

    loader = DistLinkNeighborLoader(
        data=data,
        edge_label_index=edge_label_index_train,
        num_neighbors=[-1],
        batch_size=10,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=master_port,
        current_ctx=current_ctx,
        rpc_worker_names={},
        concurrency=concurrency,
        device=device,
        drop_last=True,
        async_sampling=async_sampling
    )

    assert 'DistLinkNeighborLoader()' in str(loader)
    assert str(mp.current_process().pid) in str(loader)
    assert isinstance(loader.neighbor_sampler, DistNeighborSampler)

    for batch in loader:
        assert isinstance(batch, Data)
        assert batch.x.device == device
        assert batch.x.size(0) >= 0
        assert batch.n_id.size() == (batch.num_nodes, )
        assert batch.input_id.numel() == batch.batch_size == 10
        assert batch.edge_index.device == device
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        assert batch.edge_attr.device == device

def dist_neighbor_loader_homo(
    tmp_path: str,
    world_size: int,
    rank: int,
    master_addr: str,
    master_port: int,
    num_workers: int,
    concurrency: int,
    async_sampling: bool,
    device=torch.device('cpu')
):

    data, input_nodes = create_dist_data(tmp_path, rank)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-loader-test'
    )

    loader = DistNeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=10,
        num_workers=num_workers,
        input_nodes=input_nodes,
        master_addr=master_addr,
        master_port=master_port,
        current_ctx=current_ctx,
        rpc_worker_names={},
        concurrency=concurrency,
        device=device,
        drop_last=True,
        async_sampling=async_sampling
    )

    assert 'DistNeighborLoader()' in str(loader)
    assert str(mp.current_process().pid) in str(loader)
    assert isinstance(loader.neighbor_sampler, DistNeighborSampler)

    for batch in loader:
        assert isinstance(batch, Data)
        assert batch.x.device == device
        assert batch.x.size(0) >= 0
        assert batch.n_id.size() == (batch.num_nodes, )
        assert batch.input_id.numel() == batch.batch_size == 10
        assert batch.edge_index.device == device
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        assert batch.edge_attr.device == device
        assert batch.edge_attr.size(0) == batch.edge_index.size(1)


def dist_neighbor_loader_hetero(
    tmp_path: str,
    world_size: int,
    rank: int,
    master_addr: str,
    master_port: int,
    num_workers: int,
    concurrency: int,
    async_sampling: bool,
    device=torch.device('cpu')
):

    data, input_nodes = create_dist_data(tmp_path, rank)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-loader-test'
    )

    loader = DistNeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=10,
        num_workers=num_workers,
        input_nodes=input_nodes,
        master_addr=master_addr,
        master_port=master_port,
        current_ctx=current_ctx,
        rpc_worker_names={},
        concurrency=concurrency,
        device=device,
        drop_last=True,
        async_sampling=async_sampling
    )

    assert 'DistNeighborLoader()' in str(loader)
    assert str(mp.current_process().pid) in str(loader)
    assert isinstance(loader.neighbor_sampler, DistNeighborSampler)
    assert data[0].meta['is_hetero'] == True

    for batch in loader:
        assert isinstance(batch, HeteroData)
        assert batch['v0'].input_id.numel() == batch['v0'].batch_size == 10
        assert len(batch.node_types) == 2
        for ntype in batch.node_types:
            assert torch.equal(batch[ntype].x, batch.x_dict[ntype])
            assert batch.x_dict[ntype].device == device
            assert batch.x_dict[ntype].size(0) >= 0
            assert batch[ntype].n_id.size() == (batch[ntype].num_nodes, )
        assert len(batch.edge_types) == 4
        for etype in batch.edge_types[:2]:
            assert batch[etype].edge_index.device == device
            assert batch[etype].edge_attr.device == device
            assert batch[etype].edge_attr.size(
                0) == batch[etype].edge_index.size(1)


@onlyLinux
@pytest.mark.skipif(not WITH_METIS, reason='Not compiled with METIS support')
@pytest.mark.parametrize('num_workers', [0, 2])
@pytest.mark.parametrize('concurrency', [1, 10])
@pytest.mark.parametrize('async_sampling', [True, False])
def test_dist_neighbor_loader_homo(
        tmp_path, num_workers, concurrency, async_sampling):

    mp_context = torch.multiprocessing.get_context('spawn')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    addr = 'localhost'

    data = FakeDataset(
        num_graphs=1,
        avg_num_nodes=100,
        avg_degree=3,
        edge_dim=2)[0]

    num_parts = 2
    partitioner = Partitioner(data, num_parts, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(target=dist_neighbor_loader_homo, args=(
        tmp_path, num_parts, 0, addr, port, num_workers, concurrency, async_sampling))

    w1 = mp_context.Process(target=dist_neighbor_loader_homo, args=(
        tmp_path, num_parts, 1, addr, port, num_workers, concurrency, async_sampling))

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@onlyLinux
@pytest.mark.skipif(not WITH_METIS, reason='Not compiled with METIS support')
@pytest.mark.parametrize('num_workers', [0, 2])
@pytest.mark.parametrize('concurrency', [1, 10])
@pytest.mark.parametrize('async_sampling', [True, False])
def test_dist_link_neighbor_loader_homo(
        tmp_path, num_workers, concurrency, async_sampling):

    mp_context = torch.multiprocessing.get_context('spawn')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    addr = 'localhost'

    data = FakeDataset(
        num_graphs=1,
        avg_num_nodes=100,
        avg_degree=3,
        edge_dim=2)[0]

    num_parts = 2
    partitioner = Partitioner(data, num_parts, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(target=dist_link_neighbor_loader_homo, args=(
        tmp_path, num_parts, 0, addr, port, num_workers, concurrency, async_sampling))

    w1 = mp_context.Process(target=dist_link_neighbor_loader_homo, args=(
        tmp_path, num_parts, 1, addr, port, num_workers, concurrency, async_sampling))

    w0.start()
    w1.start()
    w0.join()
    w1.join()


# @onlyLinux
# @pytest.mark.skipif(not WITH_METIS, reason='Not compiled with METIS support')
# @pytest.mark.parametrize('num_workers', [0, 2])
# @pytest.mark.parametrize('concurrency', [1, 10])
# @pytest.mark.parametrize('async_sampling', [True, False])
# def test_dist_neighbor_loader_hetero(
#         tmp_path, num_workers, concurrency, async_sampling):

#     mp_context = torch.multiprocessing.get_context('spawn')
#     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     s.bind(('127.0.0.1', 0))
#     port = s.getsockname()[1]
#     s.close()
#     addr = 'localhost'

#     data = FakeHeteroDataset(
#         num_graphs=1,
#         avg_num_nodes=100,
#         avg_degree=3,
#         num_node_types=2,
#         num_edge_types=4,
#         edge_dim=2)[0]

#     num_parts = 2
#     partitioner = Partitioner(data, num_parts, tmp_path)
#     partitioner.generate_partition()

#     w0 = mp_context.Process(target=dist_neighbor_loader_hetero, args=(
#         tmp_path, num_parts, 0, addr, port, num_workers, concurrency, async_sampling))

#     w1 = mp_context.Process(target=dist_neighbor_loader_hetero, args=(
#         tmp_path, num_parts, 1, addr, port, num_workers, concurrency, async_sampling))

#     w0.start()
#     w1.start()
#     w0.join()
#     w1.join()
