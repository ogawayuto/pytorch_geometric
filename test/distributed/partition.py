import json
import logging
import os
import os.path as osp
from typing import List, Optional, Union, Tuple, Dict

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import ClusterData
from torch_geometric.typing import EdgeType, EdgeTypeStr, NodeType, as_str


class Partitioner:
    r"""Partition the graph structure and its features of a
    :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` object.
    Partitioned data output will be structured like this:

    **Homogeneous graphs:**

    .. code-block::

        root/
        |-- META.json
        |-- node_map.pt
        |-- edge_map.pt
        |-- part0/
            |-- graph.pt
            |-- node_feats.pt
            |-- edge_feats.pt
        |-- part1/
            |-- graph.pt
            |-- node_feats.pt
            |-- edge_feats.pt

    **Heterogeneous graphs:**

    .. code-block::

        root/
        |-- META.json
        |-- node_map/
            |-- ntype1.pt
            |-- ntype2.pt
        |-- edge_map/
            |-- etype1.pt
            |-- etype2.pt
        |-- part0/
            |-- graph.pt
            |-- node_feats.pt
            |-- edge_feats.pt
        |-- part1/
            |-- graph.pt
            |-- node_feats.pt
            |-- edge_feats.pt

    Args:
        data (Data or HeteroData): The data object.
        num_parts (int): The number of partitions.
        root (str): Root directory where the partitioned dataset should be
            saved.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_parts: int,
        root: str,
    ):
        assert num_parts > 1

        self.data = data
        self.num_parts = num_parts
        self.root = root

    @property
    def is_hetero(self) -> bool:
        return isinstance(self.data, HeteroData)

    @property
    def node_types(self) -> Optional[List[NodeType]]:
        return self.data.node_types if self.is_hetero else None

    @property
    def edge_types(self) -> Optional[List[EdgeType]]:
        return self.data.edge_types if self.is_hetero else None

    def generate_partition(self):
        r"""Generates the partition."""
        os.makedirs(self.root, exist_ok=True)

        logging.info('Saving metadata')
        meta = {
            'num_parts': self.num_parts,
            'is_hetero': self.is_hetero,
            'node_types': self.node_types,
            'edge_types': self.edge_types,
        }
        with open(osp.join(self.root, 'META.json'), 'w') as f:
            json.dump(meta, f)

        data = self.data.to_homogeneous() if self.is_hetero else self.data
        cluster_data = ClusterData(
            data,
            num_parts=self.num_parts,
            log=True,
            keep_inter_cluster_edges=True,
        )

        node_perm = cluster_data.partition.node_perm
        partptr = cluster_data.partition.partptr
        edge_perm = cluster_data.partition.edge_perm

        node_map = torch.empty(data.num_nodes, dtype=torch.int64)
        edge_map = torch.empty(data.num_edges, dtype=torch.int64)

        if self.is_hetero:
            node_offset, edge_offset = {}, {}

            offset = 0
            for node_type in self.node_types:
                node_offset[node_type] = offset
                offset += self.data[node_type].num_nodes

            offset = 0
            for edge_name in self.edge_types:
                edge_offset[edge_name] = offset
                offset += offset + self.data.num_edges_dict[edge_name]

            edge_start = 0
            for pid in range(self.num_parts):
                logging.info(f'Saving graph partition {pid}')
                path = osp.join(self.root, f'part_{pid}')
                os.makedirs(path, exist_ok=True)

                part_data = cluster_data[pid]
                start, end = int(partptr[pid]), int(partptr[pid + 1])

                num_edges = part_data.num_edges
                edge_id = edge_perm[edge_start:edge_start + num_edges]
                edge_map[edge_id] = pid
                edge_start += +num_edges

                node_id = node_perm[start:end]
                node_map[node_id] = pid

                out = {}
                for i, edge_type in enumerate(self.edge_types):
                    src, _, dst = edge_type
                    size = (self.data[src].num_nodes, self.data[dst].num_nodes)

                    mask = part_data.edge_type == i
                    out[edge_type] = {
                        'edge_id': edge_id[mask],
                        'row': part_data.edge_index[0, mask],
                        'col': part_data.edge_index[1, mask],
                        'size': size,
                    }
                torch.save(out, osp.join(path, 'graph.pt'))

                out = {}
                for i, node_type in enumerate(self.node_types):
                    mask = part_data.node_type == i
                    x = part_data.x[mask] if 'x' in part_data else None
                    out[node_type] = {
                        'global_id': node_id[mask],
                        'feats': dict(x=x),
                    }
                torch.save(out, osp.join(path, 'node_feats.pt'))

                out = {}
                for i, edge_type in enumerate(self.edge_types):
                    mask = part_data.edge_type == i
                    edge_attr = None
                    if 'edge_attr' in part_data:
                        edge_attr = part_data.edge_attr[mask]
                    out[edge_type] = {
                        'global_id': edge_id[mask],
                        'feats': dict(edge_attr=edge_attr),
                    }
                torch.save(out, osp.join(path, 'edge_feats.pt'))

            logging.info('Saving partition mapping info')

            path = osp.join(self.root, 'node_map')
            os.makedirs(path, exist_ok=True)
            for i, node_type in enumerate(self.node_types):
                mask = data.node_type == i
                torch.save(node_map[mask], osp.join(path, f'{node_type}.pt'))

            path = osp.join(self.root, 'edge_map')
            os.makedirs(path, exist_ok=True)
            for i, edge_type in enumerate(self.edge_types):
                mask = data.edge_type == i
                torch.save(edge_map[mask],
                           osp.join(path, f'{EdgeTypeStr(edge_type)}.pt'))

        else:  # `if not self.is_hetero:`

            edge_start = 0
            for pid in range(self.num_parts):
                logging.info(f'Saving graph partition {pid}')
                path = osp.join(self.root, f'part_{pid}')
                os.makedirs(path, exist_ok=True)

                part_data = cluster_data[pid]
                start, end = int(partptr[pid]), int(partptr[pid + 1])

                num_edges = part_data.num_edges
                edge_id = edge_perm[edge_start:edge_start + num_edges]
                edge_map[edge_id] = pid
                edge_start += num_edges

                node_id = node_perm[start:end]
                node_map[node_id] = pid

                torch.save(
                    {
                        'edge_id': edge_id,
                        'row': part_data.edge_index[0],
                        'col': part_data.edge_index[1],
                        'size': (data.num_nodes, data.num_nodes),
                    }, osp.join(path, 'graph.pt'))

                torch.save(
                    {
                        'global_id': node_id,
                        'feats': dict(x=part_data.x),
                    }, osp.join(path, 'node_feats.pt'))

                torch.save(
                    {
                        'global_id': edge_id,
                        'feats': dict(edge_attr=part_data.edge_attr),
                    }, osp.join(path, 'edge_feats.pt'))

            logging.info('Saving partition mapping info')
            torch.save(node_map, osp.join(self.root, 'node_map.pt'))
            torch.save(edge_map, osp.join(self.root, 'edge_map.pt'))

def load_partition_info(
    root_dir: str,
    partition_idx: int,
) -> Tuple[Dict, int, int,
    torch.Tensor,
    torch.Tensor]:

    # load the partition with PyG format (graphstore/featurestore)
    with open(os.path.join(root_dir, 'META.json'), 'rb') as infile:
        meta = json.load(infile)
    num_partitions = meta['num_parts']
    assert partition_idx >= 0
    assert partition_idx < num_partitions
    partition_dir = os.path.join(root_dir, f'part_{partition_idx}')
    assert os.path.exists(partition_dir)


    if meta['is_hetero']==False:
        node_pb = torch.load(os.path.join(root_dir, 'node_map.pt'))
        edge_pb = torch.load(os.path.join(root_dir, 'edge_map.pt'))

        return (
            meta, num_partitions, partition_idx, node_pb, edge_pb
        )
    else:
        node_pb_dict = {}
        node_pb_dir = os.path.join(root_dir, 'node_map')
        for ntype in meta['node_types']:
            node_pb_dict[ntype] = torch.load(
                os.path.join(node_pb_dir, f'{as_str(ntype)}.pt'))


        edge_pb_dict = {}
        edge_pb_dir = os.path.join(root_dir, 'edge_map')
        for etype in meta['edge_types']:
            edge_pb_dict[tuple(etype)] = torch.load(
                os.path.join(edge_pb_dir, f'{as_str(etype)}.pt'))

        return (
            meta, num_partitions, partition_idx, node_pb_dict, edge_pb_dict
        )


