from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore
from torch_geometric.sampler import SamplerOutput
from torch_geometric.typing import EdgeType, NodeType


@dataclass
class NodeDict:
    r"""Class used during heterogeneous sampling:
    1) The nodes to serve as source nodes in the next layer
    2) The nodes with duplicates that are further needed to create COO output
    3) The output nodes without duplicates
    """
    src: Dict[NodeType, Tensor] = field(default_factory=dict)
    with_dupl: Dict[NodeType, Tensor] = field(default_factory=dict)
    out: Dict[NodeType, Tensor] = field(default_factory=dict)


@dataclass
class BatchDict:
    r"""Class used during disjoint heterogeneous sampling:
    1) The batch to serve as initial subgraph IDs for source nodes in the next
       layer
    2) The subgraph IDs with duplicates that are further needed to create COO
       output
    3) The output subgraph IDs without duplicates
    """
    src: Dict[NodeType, Tensor] = field(default_factory=dict)
    with_duple: Dict[NodeType, Tensor] = field(default_factory=dict)
    out: Dict[NodeType, Tensor] = field(default_factory=dict)


def remove_duplicates(
    out: SamplerOutput,
    node: Tensor,
    batch: Optional[Tensor] = None,
    disjoint: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:

    num_nodes = node.numel()
    node_combined = torch.cat([node, out.node])

    if not disjoint:
        _, idx = np.unique(node_combined.cpu().numpy(), return_index=True)
        idx = torch.from_numpy(idx).to(node.device).sort().values

        node = node_combined[idx]
        src = node[num_nodes:]

        return (src, node, None, None)

    else:
        batch_combined = torch.cat([batch, out.batch])
        node_batch = torch.stack([batch_combined, node_combined], dim=0)

        _, idx = np.unique(node_batch.cpu().numpy(), axis=1, return_index=True)
        idx = torch.from_numpy(idx).to(node.device).sort().values

        batch = batch_combined[idx]
        node = node_combined[idx]
        src_batch = batch[num_nodes:]
        src = node[num_nodes:]

        return (src, node, src_batch, batch)


def as_str(type: Union[NodeType, EdgeType]) -> str:
    if isinstance(type, NodeType):
        return type
    elif isinstance(type, (list, tuple)) and len(type) == 3:
        return '__'.join(type)
    return ''


def reverse_edge_type(etype: EdgeType):
    src, edge, dst = etype
    if not src == dst:
        if edge.split("_",
                      1)[0] == 'rev':  # undirected edge with `rev_` prefix.
            edge = edge.split("_", 1)[1]
        else:
            edge = 'rev_' + edge
    return (dst, edge, src)
