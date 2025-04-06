import math
import numpy as np
import re
from typing import *

import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm

from tracegnn.data import TraceGraph, TraceGraphIDManager
from tracegnn.utils import ArrayBuffer
from pathlib import Path

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'Times New Roman'

__all__ = [
    'init_plot_styles',
    'plot_trace_graph',
    'plot_nx_graph',
    'plot_grid',
    'plot_latency_hist',
    'plot_latency_std_hist',
    'plot_metrics',
    'MetricPlotter',
]

# fm.FontManager.addfont('notebooks/Times New Roman.ttf')


def init_plot_styles():
    sns.set(style='whitegrid')


def plot_trace_graph(ax, g: TraceGraph, id_manager: TraceGraphIDManager):
    def nudge(pos, x_shift, y_shift):
        return {n: (x + x_shift, y + y_shift) for n, (x, y) in pos.items()}

    def format_label(d):
        op_name = d['operation']
        span_count = d['span_count']
        if round(span_count) == int(span_count):
            span_count = int(span_count)

        if span_count == 1:
            return f'{op_name}\nlatency: {d["avg_latency"]}'
        else:
            return (
                f'{op_name} ({span_count})\n'
                f'latency: avg={d["avg_latency"]:.3f} max={d["max_latency"]:.3f} min={d["min_latency"]:.3f}'
            )

    g = g.networkx_graph(id_manager)
    node_labels = {
        i: format_label(d)
        for i, d in g.nodes.data()
    }

    pos = nx.spring_layout(g)
    pos_nodes = nudge(pos, 0, 0.1)

    num_node_types = id_manager.num_operations
    cmap = cm.get_cmap('hsv', num_node_types + 1)
    node_colors = [
        cmap(d['node_type'] * 131 % (num_node_types + 1))
        for i, d in g.nodes.data()
    ]

    nx.draw_networkx(
        g, pos=pos, ax=ax, with_labels=False,
        node_size=1000, node_color=node_colors,
        font_size=14, width=1.5,
    )
    nx.draw_networkx_labels(g, labels=node_labels, pos=pos_nodes, ax=ax)


def plot_nx_graph(ax, g: nx.Graph, id_manager: TraceGraphIDManager):
    def nudge(pos, x_shift, y_shift):
        return {n: (x + x_shift, y + y_shift) for n, (x, y) in pos.items()}

    node_labels = {
        i: f'{d["operation"]}'
        for i, d in g.nodes.data()
    }

    pos = nx.spring_layout(g)
    pos_nodes = nudge(pos, 0, 0.1)

    num_node_types = id_manager.num_operations
    cmap = cm.get_cmap('hsv', num_node_types + 1)
    node_colors = [
        cmap(d['node_type'] * 131 % (num_node_types + 1))
        for i, d in g.nodes.data()
    ]
    cmap_edge = cm.get_cmap('binary')
    edge_color = [cmap_edge(i[2]['weight']) for i in list(g.edges.data())]

    nx.draw_networkx(
        g, pos=pos, ax=ax, with_labels=False,
        node_size=100, node_color=node_colors,
        font_size=14, width=1.5, edge_color=edge_color
    )


def plot_grid(plot_fn,
              data_list: Sequence[Any],
              n_cols: Optional[int] = None,
              ax_size: Tuple[int, int] = (10, 10),
              ax_pad: float = 1.0,
              **kwargs) -> plt.Figure:
    # determine the figure size
    if n_cols is None:
        n_cols = math.ceil(math.sqrt(len(data_list)))
    n_rows = (len(data_list) + n_cols - 1) // n_cols

    # create the figure
    w, h = ax_size
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(h * n_rows, w * n_cols))

    if n_rows > 1:
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_cols + j
                if idx < len(data_list):
                    plot_fn(ax[i, j], data_list[idx], **kwargs)
    else:
        for j in range(n_cols):
            plot_fn(ax[j], data_list[j], **kwargs)

    # configure the figure
    fig.tight_layout(pad=ax_pad)

    return fig


def plot_latency_hist(latency_dict,
                      id_manager,
                      ax_size: Tuple[int, int] = (12, 3),
                      output_file: Optional[str] = None
                      ):
    # plot
    N = len(latency_dict)
    fig, ax = plt.subplots(N, 1, figsize=(ax_size[0], ax_size[1] * N))
    for i, key in enumerate(sorted(latency_dict)):
        values = np.asarray(latency_dict[key])
        if len(values) > 1:
            values = values[values >= 0]
            values = values[values <= np.percentile(values, 99)]
        bins = len(set(values))
        kw = {}
        if bins >= 300:
            kw['bins'] = 100
        sns.histplot(values, kde=False, ax=ax[i], **kw)
        ax[i].set_title(id_manager.operation_id.reverse_map(key))

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
    else:
        return fig


def plot_latency_std_hist(latency_std_dict,
                          ax_size: Tuple[int, int] = (12, 3),
                          std_limit: Tuple[float, float] = (0, 6),
                          output_file: Optional[str] = None,
                          ):
    # assemble data
    def F(arr):
        if isinstance(arr, ArrayBuffer):
            arr = arr.array
        return arr[np.isfinite(arr)]

    def key_map(k):
        if 'drop' in k:
            return 'Structure Anomalies'
        elif 'latency' in k:
            return 'Time Anomalies'
        elif 'train' in k:
            return 'Train Normal'
        elif 'val' in k:
            return 'Validation Normal'
        else:
            return 'Test Normal'

    # Set Font
    # fpath = Path('/root/tracegnn/notebooks/Times New Roman.ttf')

    latency_std_dict = {key_map(k): F(v) for k, v in latency_std_dict.items()}
    latency_std_dict = {k: v for k, v in latency_std_dict.items() if len(v) > 0}

    print(latency_std_dict)
    if not latency_std_dict:
        return None

    idx = np.arange(max(len(a) for a in latency_std_dict.values()))
    cols = {
        key: pd.Series(val, index=idx[:len(val)], name=key)
        for key, val in latency_std_dict.items()
    }
    K = len(cols)
    data = pd.DataFrame(cols)

    data = data[['Train Normal', 'Validation Normal', 'Test Normal', 'Structure Anomalies', 'Time Anomalies']]

    # plot
    fig, ax = plt.subplots(3, 2, figsize=(ax_size[0], ax_size[1] * 3))
    ax: List[plt.Axes] = ax.reshape(-1)
    print(ax.shape)

    # cdf
    sns.ecdfplot(data, ax=ax[0], legend=True)
    ax[0].set_ylabel('CDF', fontsize=14, font=fpath)
    ax[0].set_xlabel('Gaussian Std', fontsize=14, font=fpath)
    ax[0].set_title('CDF', font=fpath, size=16)
    ax[0].set_xlim(std_limit)
    plt.setp(ax[0].get_xticklabels(), fontsize=14, font=fpath)
    plt.setp(ax[0].get_yticklabels(), fontsize=14, font=fpath)
    print(ax[0].get_legend().get_texts())
    # ax[0].add_legend(label_order = ['Train Normal', 'Validation Normal', 'Test Normal', 'Structure Anomalies', 'Time Anomalies'])
    plt.setp(ax[0].get_legend().get_texts(), fontsize=14, font=fpath)

    # density
    bin_count = 100
    densities = {}

    col_list = ['Train Normal', 'Validation Normal', 'Test Normal', 'Structure Anomalies', 'Time Anomalies']

    for i, key in enumerate(col_list, 1):
        bin_values, bin_edges = np.asarray(np.histogram(data[key], range=std_limit, bins=bin_count))
        total_count = len(data[key].dropna())
        densities[key] = x, y, w = (bin_edges[:-1], bin_values / total_count, bin_edges[-1] - bin_edges[-2])
        ax[i].bar(x, y, width=w, align='edge', label=key)
        ax[i].set_title(key, font=fpath, size=16)
        ax[i].set_xlim(std_limit)
        ax[i].set_ylabel('Proportion', fontsize=14, font=fpath)
        ax[i].set_xlabel('Gaussian Std', fontsize=14, font=fpath)
        plt.setp(ax[i].get_xticklabels(), fontsize=14, font=fpath)
        plt.setp(ax[i].get_yticklabels(), fontsize=14, font=fpath)

        # ax[i].legend(prop=fm.FontProperties(fname=Path('/root/tracegnn/notebooks/Times New Roman.ttf'), size=14))

    # # density merged
    # for key, (x, y, w) in densities.items():
    #     ax[1].bar(x, y, width=w, align='edge', label=key, alpha=.6)
    #     ax[1].set_title(key)
    #     ax[1].set_xlim(std_limit)
    #     ax[1].legend()

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
    else:
        return fig


def plot_p_node_edge_hist(p_node_count,
                          p_edge,
                          ax_size: Tuple[int, int] = (12, 3),
                          std_limit: Tuple[float, float] = (0, 1),
                          output_file: Optional[str] = None,
                          ):
    # assemble data
    def F(arr):
        if isinstance(arr, ArrayBuffer):
            arr = arr.array
        return arr[np.isfinite(arr)]

    def key_map(k):
        if 'drop' in k:
            return 'Structure Anomalies'
        elif 'latency' in k:
            return 'Time Anomalies'
        elif 'train' in k:
            return 'Train Normal'
        elif 'val' in k:
            return 'Validation Normal'
        else:
            return 'Test Normal'

    # Set Font
    # fpath = Path('/root/tracegnn/notebooks/Times New Roman.ttf')

    # Load node count data
    p_node_count = {key_map(k): F(v) for k, v in p_node_count.items()}
    p_node_count = {k: v for k, v in p_node_count.items() if len(v) > 0}
    if not p_node_count:
        return None

    idx = np.arange(max(len(a) for a in p_node_count.values()))
    cols = {
        key: pd.Series(val, index=idx[:len(val)], name=key)
        for key, val in p_node_count.items()
    }
    node_data = pd.DataFrame(cols)
    node_data = node_data[['Test Normal', 'Structure Anomalies']]

    # Load edge data
    p_edge = {key_map(k): F(v) for k, v in p_edge.items()}
    p_edge = {k: v for k, v in p_edge.items() if len(v) > 0}
    if not p_edge:
        return None

    idx = np.arange(max(len(a) for a in p_edge.values()))
    cols = {
        key: pd.Series(val, index=idx[:len(val)], name=key)
        for key, val in p_edge.items()
    }
    edge_data = pd.DataFrame(cols)
    edge_data = edge_data[['Test Normal', 'Structure Anomalies']]

    # plot
    fig, ax = plt.subplots(3, 2, figsize=(ax_size[0], ax_size[1] * 3))
    ax: List[plt.Axes] = ax.reshape(-1)

    # density
    bin_count = 50
    densities = {}

    # cdf (p(N))
    sns.ecdfplot(node_data, ax=ax[0], legend=True)
    ax[0].set_ylabel('CDF', fontsize=14, font=fpath)
    ax[0].set_xlabel('$p(N|\mathbf{z})$', fontsize=14)
    ax[0].set_title('CDF of $p(N|\mathbf{z})$', font=fpath, size=16)
    ax[0].set_xlim(std_limit)
    ax[0].set_yscale("log")
    plt.setp(ax[0].get_xticklabels(), fontsize=14, font=fpath)
    plt.setp(ax[0].get_yticklabels(), fontsize=14, font=fpath)
    plt.setp(ax[0].get_legend().get_texts(), fontsize=14, font=fpath)

    # cdf (p(edge))
    sns.ecdfplot(edge_data, ax=ax[1], legend=True)
    ax[1].set_ylabel('CDF', fontsize=14, font=fpath)
    ax[1].set_xlabel('$p(A_{ij}|\mathbf{z})$', fontsize=14)
    ax[1].set_title('CDF of $p(A_{ij}|\mathbf{z})$', font=fpath, size=16)
    ax[1].set_xlim(std_limit)
    ax[1].set_yscale("log")
    plt.setp(ax[1].get_xticklabels(), fontsize=14, font=fpath)
    plt.setp(ax[1].get_yticklabels(), fontsize=14, font=fpath)
    plt.setp(ax[1].get_legend().get_texts(), fontsize=14, font=fpath)

    col_list = ['Test Normal', 'Structure Anomalies']

    for i, key in enumerate(col_list, 1):
        bin_values, bin_edges = np.asarray(np.histogram(node_data[key], range=std_limit, bins=bin_count))
        total_count = len(node_data[key].dropna())
        densities[key] = x, y, w = (bin_edges[:-1], bin_values / total_count, bin_edges[-1] - bin_edges[-2])
        ax[2*i].bar(x, y, width=w, align='edge', label=key)
        ax[2*i].set_title(key, font=fpath, size=16)
        ax[2*i].set_xlim(std_limit)
        ax[2*i].set_yscale("log")
        ax[2*i].set_ylabel('Proportion', fontsize=14, font=fpath)
        ax[2*i].set_xlabel('$p(N|\mathbf{z})$', fontsize=14)
        plt.setp(ax[2*i].get_xticklabels(), fontsize=14, font=fpath)
        plt.setp(ax[2*i].get_yticklabels(), fontsize=14, font=fpath)

        bin_values, bin_edges = np.asarray(np.histogram(edge_data[key], range=std_limit, bins=bin_count))
        total_count = len(edge_data[key].dropna())
        densities[key] = x, y, w = (bin_edges[:-1], bin_values / total_count, bin_edges[-1] - bin_edges[-2])
        ax[2*i+1].bar(x, y, width=w, align='edge', label=key)
        ax[2*i+1].set_title(key, font=fpath, size=16)
        ax[2*i+1].set_xlim(std_limit)
        ax[2*i+1].set_yscale("log")
        ax[2*i+1].set_ylabel('Proportion', fontsize=14, font=fpath)
        ax[2*i+1].set_xlabel('$p(A_{ij}|\mathbf{z})$', fontsize=14)
        plt.setp(ax[2*i+1].get_xticklabels(), fontsize=14, font=fpath)
        plt.setp(ax[2*i+1].get_yticklabels(), fontsize=14, font=fpath)

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
    else:
        return fig


def plot_metrics(index,
                 metrics: Sequence[Dict[str, float]],
                 title: Optional[str] = None,
                 keys: Optional[Union[Sequence[str], str, re.Pattern]] = None,
                 x_label: Optional[str] = None,
                 y_label: Optional[str] = None,
                 y_limit: Optional[Tuple[float, float]] = None,
                 figsize: Tuple[int, int] = (12, 6),
                 output_file: Optional[str] = None,
                 ) -> Optional[plt.Figure]:
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # gather keys
    exist_keys = {}
    for row in metrics:
        for k in row:
            if k not in exist_keys:
                exist_keys[k] = 0
            exist_keys[k] += 1

    if isinstance(keys, str):
        keys = re.compile(keys)
    if hasattr(keys, 'match') or keys is None:
        if keys is None:
            keys = list(exist_keys)
        else:
            keys = [k for k in exist_keys if keys.match(k)]
    else:
        keys = [k for k in keys if k in keys]
    keys.sort(key=lambda k: (-exist_keys[k], k))

    # now plot the metrics
    N = len(metrics)
    y_range = None
    # min_value, low_value, high_value, max_value = None, None, None, None

    cmap = plt.cm.plasma

    for i, key in enumerate(keys):
        key_idx, key_val = [], []
        for row, idx in zip(metrics, index):
            if key in row:
                key_idx.append(idx)
                key_val.append(row[key])
        if key_idx:
            key_idx = np.array(key_idx)
            key_val = np.array(key_val)
            key_val_p = key_val[np.isfinite(key_val)]

            key_min, key_low, key_high, key_max = np.percentile(key_val_p, [0, 20, 80, 100])
            # if min_value is None or key_min < min_value:
            #     min_value = key_min
            # if low_value is None or key_low < low_value:
            #     low_value = key_low
            # if high_value is None or key_high > high_value:
            #     high_value = key_high
            # if max_value is None or key_max > max_value:
            #     max_value = key_max

            kw = {
                'label': key,
                'alpha': .9,
                # 'color': cmap(i * (cmap.N - 1) / len(keys)),
            }
            if len(key_idx) < N * 0.1 and len(key_idx) < 150:
                kw['marker'] = 'x'
            plt.plot(key_idx, key_val, **kw)

            # infer y_limit
            key_range = key_high - key_low
            key_range = (
                max(key_min - key_range * 0.1, key_low - key_range * 5),
                min(key_max + key_range * 0.1, key_high + key_range * 5),
            )

            if y_range is None:
                y_range = key_range
            else:
                y_range = (
                    min(key_range[0], y_range[0]),
                    max(key_range[1], y_range[1]),
                )

            # if high_value is not None:  # which implies the other three
            #     y_range = high_value - low_value
            #     if y_limit is None:
            #         y_limit = [
            #             max(min_value - y_range * 0.1, low_value - y_range * 5),
            #             min(max_value + y_range * 0.1, high_value + y_range * 5),
            #         ]

    if y_limit is None:
        y_limit = y_range

    # configure the figure
    if title:
        plt.title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if y_limit and all(math.isfinite(v) for v in y_limit):
        plt.ylim(y_limit)

    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
    else:
        return fig


class MetricPlotter(object):

    def __init__(self, initial_step: int = 1):
        self.step = initial_step
        self.index = []
        self.data = []

    def inc_step(self):
        self.step += 1

    def add(self, metrics=None, step=None, **kwargs):
        d = dict(metrics or ())
        d.update(kwargs)
        if step is None:
            step = self.step
        if not self.index or self.index[-1] != step:
            self.index.append(step)
            self.data.append(d)
        else:
            self.data[-1].update(d)

    def plot(self, **kwargs):
        return plot_metrics(self.index, self.data, **kwargs)
