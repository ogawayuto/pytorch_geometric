import atexit
import socket

import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore
from torch_geometric.distributed.dist_context import DistContext
from torch_geometric.distributed.dist_neighbor_sampler import (
    DistNeighborSampler,
    close_sampler,
)
from torch_geometric.distributed.rpc import init_rpc
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput
from torch_geometric.testing import withPackage


def create_data(rank, world_size):
    # create dist data
    if rank == 0:
        # partition 0
        node_id = torch.tensor([0, 1, 2, 3, 4, 5, 6, 10])
        edge_index = torch.tensor([
            [1, 2, 3, 4, 5, 0, 0],
            [0, 1, 2, 3, 4, 4, 9],
        ])
    else:
        # partition 1
        node_id = torch.tensor([0, 4, 5, 6, 7, 8, 9, 10])
        edge_index = torch.tensor([
            [5, 6, 7, 8, 9, 0, 5],
            [4, 5, 6, 7, 8, 9, 9],
        ])

    feature_store = LocalFeatureStore.from_data(node_id)
    graph_store = LocalGraphStore.from_data(None, edge_index, num_nodes=11)

    graph_store.node_pb = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    graph_store.meta.update({'num_parts': 2})
    graph_store.partition_idx = rank
    graph_store.num_partitions = world_size

    dist_data = (feature_store, graph_store)

    # create reference data
    edge_index = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0, 6, 7, 8, 9, 5],
        [0, 1, 2, 3, 4, 4, 9, 5, 6, 7, 8, 9],
    ])
    data = Data(x=None, y=None, edge_index=edge_index, num_nodes=11)

    return (dist_data, data)


def dist_neighbor_sampler_homo(
    world_size: int,
    rank: int,
    master_port: int,
    disjoint: bool = False,
):
    dist_data, data = create_data(rank, world_size)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name="dist-sampler-test",
    )

    # Initialize training process group of PyTorch.
    torch.distributed.init_process_group(
        backend="gloo",
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method="tcp://{}:{}".format('localhost', master_port),
    )

    num_neighbors = [-1, -1]
    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=num_neighbors,
        shuffle=False,
        disjoint=disjoint,
    )

    init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names={},
        master_addr='localhost',
        master_port=master_port,
    )

    dist_sampler.register_sampler_rpc()
    dist_sampler.init_event_loop()

    # close RPC & worker group at exit:
    atexit.register(close_sampler, 0, dist_sampler)
    torch.distributed.barrier()

    # seed nodes
    if rank == 0:
        input_node = torch.tensor([1, 6], dtype=torch.int64)
    else:
        input_node = torch.tensor([4, 9], dtype=torch.int64)

    inputs = NodeSamplerInput(
        input_id=None,
        node=input_node,
    )

    # evaluate distributed node sample function
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))

    torch.distributed.barrier()

    sampler = NeighborSampler(data=data, num_neighbors=num_neighbors,
                              disjoint=disjoint)

    # evaluate node sample function
    out = sampler._sample(input_node)

    # compare distributed output with single machine output
    assert torch.equal(out_dist.node, out.node)
    assert torch.equal(out_dist.row, out.row)
    assert torch.equal(out_dist.col, out.col)
    if disjoint:
        assert torch.equal(out_dist.batch, out.batch)
    assert out_dist.num_sampled_nodes == out.num_sampled_nodes
    assert out_dist.num_sampled_edges == out.num_sampled_edges

    torch.distributed.barrier()


@withPackage('pyg_lib')
@pytest.mark.parametrize("disjoint", [True, False])
def test_dist_neighbor_sampler_homo(disjoint):
    mp_context = torch.multiprocessing.get_context("spawn")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    world_size = 2
    w0 = mp_context.Process(
        target=dist_neighbor_sampler_homo,
        args=(world_size, 0, port, disjoint),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_homo,
        args=(world_size, 1, port, disjoint),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()
