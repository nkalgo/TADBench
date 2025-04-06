"""Render graph as HTML."""
import gzip
import json
import os
from typing import *

import jinja2
import networkx as nx

from tracegnn.data import *
import matplotlib as mpl
from matplotlib import cm, colors

__all__ = [
    'render_trace_graph_html',
]


def render_trace_graph_html(graph_list: Sequence[Union[TraceGraph, nx.Graph, Tuple[Any, Union[TraceGraph, nx.Graph]]]],
                            id_manager: TraceGraphIDManager,
                            output_file: Optional[str] = None,
                            page_title: Optional[str] = None,
                            graph_size: Tuple[int, int] = (400, 400),
                            cdn: Optional[Union[bool, str]] = True,
                            ) -> Optional[str]:
    MAX_EDGE_OPACITY = 0.9
    MIN_NODE_OPACITY = 0.1

    # compose the data array
    def make_node(i, nd):
        name = str(i)
        value = ''

        if 'span_count' in nd:
            span_count = nd['span_count']
            if round(span_count) == int(span_count):
                span_count = int(span_count)
        else:
            span_count = 1

        # name
        if 'operation' in nd:
            name = nd['operation']
            if span_count != 1:
                name = f'{name} ({span_count})'

        # value
        if 'avg_latency' in nd:
            if span_count == 1:
                value = f'latency: {nd["avg_latency"]:.3f}'
            else:
                value = f'latency: avg={nd["avg_latency"]:.3f} max={nd["max_latency"]:.3f} min={nd["min_latency"]:.3f}'

        # category
        if 'node_type' in nd:
            category = int(nd['node_type'])
        else:
            category = 0

        return {
            'name': str(i),
            'category': category,
            'value': [name, value],
        }

    def make_tree(pos: int, pa: Optional[int], g: nx.Graph, id_manager: TraceGraphIDManager):
        cmap = cm.get_cmap('hsv', id_manager.num_operations)
        nd = g.nodes[pos]

        # value
        if 'span_count' in nd:
            span_count = nd['span_count']
            if round(span_count) == int(span_count):
                span_count = int(span_count)
        else:
            span_count = 1

        # name
        if 'operation' in nd:
            name = nd['operation']
            if span_count != 1:
                name = f'{name} ({span_count})'

        # value
        if 'avg_latency' in nd:
            if span_count == 1:
                value = f'latency: {nd["avg_latency"]:.3f}'
            else:
                value = f'latency: avg={nd["avg_latency"]:.3f} max={nd["max_latency"]:.3f} min={nd["min_latency"]:.3f}'

        # score
        if 'avg_latency_nstd' in nd:
            value += f' nstd: {nd["avg_latency_nstd"]:.3f}'

        result_dict = {}
        result_dict['name'] = str(nd['node_type'])
        result_dict['value'] = [name, value]
        result_dict['itemStyle'] = {
            'color': colors.rgb2hex(cmap(nd['node_type'])),
            'borderWidth': 5,
        }

        for nxt in sorted(g.neighbors(pos), key=lambda x: g.nodes[x]['node_type']):
            if nxt == pa: continue

            if 'children' not in result_dict:
                result_dict['children'] = []
            result_dict['children'].append(make_tree(nxt, pos, g, id_manager))

        return result_dict

    def make_edge(u, v, edge):
        r = {
            'source': int(u),
            'target': int(v),
        }
        if 'weight' in edge:
            r['lineStyle'] = {
                'opacity': edge['weight'] * MAX_EDGE_OPACITY
            }
        return r

    data = []
    for item in graph_list:
        if isinstance(item, tuple):
            graph_id, graph = item
        else:
            graph_id, graph = None, item

        if isinstance(graph, TraceGraph):
            g: nx.Graph = graph.networkx_graph(id_manager)
        elif isinstance(graph, nx.Graph):
            g: nx.Graph = graph
        else:
            raise TypeError(f'`graph` is neither a TraceGraph nor an nx.Graph: {graph!r}')

        if graph_id is not None:
            if isinstance(graph_id, int):
                graph_id = f'#{graph_id}'
            g_title = {'text': str(graph_id)}
        else:
            g_title = None

        # Check circle in graph
        if len(nx.cycle_basis(g, 0)) > 0:
            # Use graph model
            data.append({
                'title': g_title,
                'tooltip': {},
                'animationDurationUpdate': 1500,
                'animationEasingUpdate': 'quinticInOut',
                'roam': True,
                'graphData': dict(g.graph),
                'series': [
                    {
                        'type': 'graph',
                        'layout': 'force',
                        'symbolSize': 30,
                        'roam': True,
                        'tooltip': {
                            'position': 'top',
                        },
                        'label': {
                            'show': False,
                        },
                        'force': {
                            'edgeLength': 100,
                            'repulsion': 100,
                            'gravity': 0.0
                        },
                        'edgeSymbol': ['circle', 'none'],
                        'edgeSymbolSize': [4, 10],
                        'edgeLabel': {
                            'fontSize': 20
                        },
                        'data': [
                            make_node(i, g.nodes[i])
                            for i in g.nodes
                        ],
                        'links': [
                            make_edge(u, v, edge)
                            for u, v, edge in g.edges.data()
                        ],
                        'categories': [
                            {'name': id_manager.operation_id.reverse_map(i)}
                            for i in range(id_manager.num_operations)
                        ],
                        'lineStyle': {
                            'opacity': 0.9,
                            'width': 2,
                            'curveness': 0,
                        }
                    }
                ]
            })
        else:
            # Use tree model
            data.append({
                'title': g_title,
                'tooltip': {
                    'trigger': 'item',
                    'triggerOn': 'mousemove'
                },
                'graphData': dict(g.graph),
                'series': [
                    {
                        'type': 'tree',
                        'data': [make_tree(0, None, g, id_manager)],
                        'top': '10%',
                        'left': '8%',
                        'bottom': '22%',
                        'right': '20%',
                        'symbolSize': 12,
                        'edgeShape': 'polyline',
                        'edgeForkPosition': '63%',
                        'initialTreeDepth': 8,
                        'lineStyle': {
                            'width': 2
                        },
                        'tooltip': {
                            'position': 'top',
                        },
                        'label': {
                            'backgroundColor': '#fff',
                            'position': 'left',
                            'verticalAlign': 'middle',
                            'align': 'right'
                        },
                        'leaves': {
                            'label': {
                                'position': 'right',
                                'verticalAlign': 'middle',
                                'align': 'left'
                            }
                        },
                        'emphasis': {
                            'focus': 'descendant'
                        },
                        'expandAndCollapse': True,
                        'animationDuration': 550,
                        'animationDurationUpdate': 750
                    }
                ]
            })

    # render the html
    assets_dir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        'assets'
    )
    if not cdn:
        with open(os.path.join(assets_dir, 'js/echarts.min.js'), 'r', encoding='utf-8') as f:
            echarts = f.read()
    else:
        if isinstance(cdn, bool):
            cdn = None  # let the template to choose a cdn
        echarts = None
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader([assets_dir])
    )
    html = env.get_template('graph_html/render_trace_graph.html').render(
        echarts=echarts,
        cdn=cdn,
        data_json=json.dumps(data),
        page_title=page_title,
        graph_size=graph_size,
    )

    # save the file
    if output_file is not None:
        cnt = html.encode('utf-8')
        if output_file.endswith('.gz'):
            cnt = gzip.compress(cnt)
        with open(output_file, 'wb') as f:
            f.write(cnt)
    else:
        return html
