
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.distributed import (
    Partitioner,
)
import torch_geometric.transforms as T

data_path="/home/pyg/graphlearn-dev/partition_fake"

data = FakeHeteroDataset(
    num_graphs=1,
    avg_num_nodes=100,
    avg_degree=3,
    num_node_types=2,
    num_edge_types=4,
    edge_dim=2)[0]

num_parts = 2
partitioner = Partitioner(data, num_parts, data_path)
partitioner.generate_partition()

data_path = "/home/pyg/graphlearn-dev/partition_fake_und"

data = FakeHeteroDataset(
    num_graphs=1,
    avg_num_nodes=100,
    avg_degree=3,
    num_node_types=2,
    num_edge_types=4,
    edge_dim=2,
    transform=T.ToUndirected())[0]

num_parts = 2
partitioner = Partitioner(data, num_parts, data_path)
partitioner.generate_partition()