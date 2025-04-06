from typing import *

import numpy as np

from tracegnn.utils import TraceGraphLatencyRangeFile
from .trace_graph import *

__all__ = ['trace_graph_mutate']


def trace_graph_mutate(g: TraceGraph,
                       latency_range: TraceGraphLatencyRangeFile,
                       drop_ratio: Optional[Tuple[float, float]],
                       latency_ratio: Optional[Tuple[float, float]],
                       latency_delta: Optional[Tuple[float, float]],
                       latency_delta_nstd: Optional[Tuple[float, float]],
                       latency_delta_np99: Optional[Tuple[float, float]],
                       just_one_node: bool = False,
                       avg_as_whole: bool = False,
                       set_anomaly_flag: bool = True,
                       drop_subtract_latency: bool = False,
                       latency_p99_min: Optional[float] = None,
                       ):
    g_old = g

    # flag to indicate whether the graph is mutated
    is_mutated = True

    # first, gather all nodes
    g = g.deepcopy()
    nodes: List[Optional[TraceGraphNode]] = [None] * g.node_count
    parent_map: Dict[TraceGraphNode, TraceGraphNode] = {}

    for depth, idx, node, parent in g.iter_bfs(with_parent=True):
        nodes[node.node_id] = node
        parent_map[node] = parent

    # drop random nodes
    keep_mask = np.full([g.node_count], True, dtype=np.bool)
    if drop_ratio:
        def apply_latency(node, inc):
            node.features.avg_latency = max(node.features.avg_latency - inc, 0)
            node.features.max_latency = max(node.features.max_latency - inc, 0)
            node.features.min_latency = max(node.features.min_latency - inc, 0)

        indices = np.arange(1, g.node_count)
        ratio = np.random.uniform(*drop_ratio)
        drop_count = min(max(round(len(indices) * ratio), 1), len(indices))
        np.random.shuffle(indices)
        for idx in indices:
            if not keep_mask[idx]:
                continue

            for _, node in nodes[idx].iter_bfs():
                keep_mask[node.node_id] = False
                inc = node.features.avg_latency
                parent_map[node].children.remove(node)
                is_mutated = True
                drop_count -= 1

                if just_one_node:
                    drop_count = -1

                if drop_subtract_latency:
                    while node is not None:
                        apply_latency(node, inc)
                        is_mutated = True
                        node = parent_map.get(node)

            if drop_count <= 0:
                break

    # inc node latency
    if latency_ratio:
        def apply_latency(node, inc):
            if avg_as_whole:
                r = 1.
            else:
                r = 1 / node.features.span_count
            node.features.avg_latency += inc * r
            node.features.max_latency += inc
            if node.features.span_count == 1:
                node.features.min_latency += inc
            node.anomaly = 2

        indices = np.arange(1, g.node_count)[keep_mask[1:]]
        ratio = np.random.uniform(*latency_ratio)
        latency_count = min(max(round(len(indices) * ratio), 1), len(indices))
        np.random.shuffle(indices)
        for idx in indices:
            node = nodes[idx]

            if latency_delta is not None:
                inc = np.random.uniform(*latency_delta)
            elif latency_delta_nstd is not None:
                mu, std = latency_range[node.operation_id]
                inc = max(0, mu + np.random.uniform(*latency_delta_nstd) * std - node.features.avg_latency)
            elif latency_delta_np99 is not None:
                p99 = latency_range.get_item(node.operation_id)['p99']
                if latency_p99_min:
                    p99 = max(latency_p99_min, p99)
                inc = max(0, p99 * np.random.uniform(*latency_delta_np99) - node.features.avg_latency)
            else:
                raise ValueError(f'At least one of `latency_delta`, `latency_delta_nstd` and '
                                 f'`latency_delta_p99` should be specified.')

            while node is not None:
                apply_latency(node, inc)
                is_mutated = True
                latency_count -= 1

                if just_one_node:
                    latency_count = -1

                node = parent_map.get(node)

            if latency_count <= 0:
                break

    # finally, update the graph node id
    if is_mutated:
        g.assign_node_id()
        if (g.node_count == g_old.node_count) and (g.root.features.avg_latency == g_old.root.features.avg_latency):
            return g_old
        if set_anomaly_flag:
            g.data['is_anomaly'] = True
        return g
    return g_old
