import socket
from typing import Dict, List

import torch

import torch_geometric.distributed.rpc as rpc
from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.distributed.rpc import RPCRouter
from torch_geometric.testing import onlyLinux

import torch_geometric.distributed as pyg_dist

import os.path as osp

import pytest
import torch

from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.typing import EdgeTypeStr
from torch_geometric.distributed.partition import load_partition_info



def run_rpc_feature_test(
    world_size: int,
    rank: int,
    root_dir: str,
    master_port: int,
):
    print(f"------------ rank={rank}, root_dir={root_dir} ")

    graph = LocalGraphStore.from_partition(root_dir, rank)
    print(f"-------- graph={graph} ")
    edge_attrs = graph.get_all_edge_attrs()[0]
    print(f"-----777---- edge_attrs ={edge_attrs}")

    feature = LocalFeatureStore.from_partition(root_dir, rank)
    print(f"-------- feature={feature} ")

    (
        meta, num_partitions, partition_idx, node_pb, edge_pb
    ) = load_partition_info(root_dir, rank)

    #print(f"----- meta={meta}, node_pb={node_pb}, edge_pb={edge_pb}, partition_idx={partition_idx} ")

    graph.num_partitions = world_size
    graph.partition_idx = rank
    graph.node_pb = node_pb
    graph.edge_pb = edge_pb
    graph.meta = meta

    graph.labels = torch.arange(edge_attrs.size[0])

    #print(f"------8888 graph.labels={graph.labels}, edge_attrs.size[0]={edge_attrs.size[0]}  ")

    feature.num_partitions = world_size
    feature.partition_idx = rank
    feature.node_feat_pb = node_pb 
    feature.meta = meta
    #feature.set_rpc_router(rpc_router)

    partition_data = (graph, feature)


    # 1) Initialize the context info:
    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-feature-test',
    )
    rpc_worker_names: Dict[DistRole, List[str]] = {}

    print(f"----------- 222 -------------\n\n\n ")

    batch_size = 256

    # Create distributed neighbor loader for training
    train_idx = torch.arange(batch_size)  #+ 128 * rank
    #train_idx = train_idx.split(train_idx.size(0) // num_training_procs_per_node)[local_proc_rank]
    num_workers=2
    
    loader = pyg_dist.DistNeighborLoader(
    data=partition_data,
    num_neighbors=[15, 10, 5],
    input_nodes=train_idx,
    batch_size=batch_size,
    shuffle=False,
    collect_features=True,
    device=torch.device('cpu'),
    num_workers=num_workers,
    concurrency=2,
    master_addr='127.0.0.1',
    master_port=master_port,
    async_sampling = True,
    filter_per_worker = False,
    current_ctx=current_ctx,
    rpc_worker_names=rpc_worker_names
    )

    for i, batch in enumerate(loader):
        #if i>2:
        print(f"\n\n\n\n------ZZZZZZZZZZZZZZZ  -------i={i}, batch={batch} \n\n\n")

    print("---------- done ------------ ")
    rpc.shutdown_rpc()

@onlyLinux
def test_dist_feature_lookup():

    data = FakeDataset()[0]
    num_parts = 2
    root_dir = "./partition"

    partitioner = Partitioner(data, num_parts, root_dir)
    partitioner.generate_partition()


    mp_context = torch.multiprocessing.get_context('spawn')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()

    w0 = mp_context.Process(target=run_rpc_feature_test,
                            args=(2, 0, root_dir, port))
    w1 = mp_context.Process(target=run_rpc_feature_test,
                            args=(2, 1, root_dir, port))

    w0.start()
    w1.start()
    w0.join()
    w1.join()

if __name__ == '__main__':
    test_dist_feature_lookup()
