import re

import pandas as pd
import seaborn as sns
from tqdm import tqdm

from tracegnn.data import *
from tracegnn.visualize import *


@click.group()
def main():
    pass


# def open_db(input_dir, root_dir, names):
#     input_dir = os.path.abspath(input_dir)
#     if not os.path.isdir(input_dir):
#         raise IOError(f'`--input_dir` not exist: {input_dir}')
#
#     if names:
#         names = set(sum([n.split(',') for n in names], []))
#
#     if os.path.exists(os.path.join(input_dir, 'service_id.yml')):
#         root_dir = input_dir
#         input_dirs = [
#             os.path.join(input_dir, name)
#             for name in os.listdir(input_dir)
#             if (
#                     (not names and re.match(r'^(\d{4}-\d{2}-\d{2}|train|val|l?test.*)$', name)) or
#                     (name in names)
#             )
#         ]
#     else:
#         input_dirs = input_dir
#
#     if root_dir is None:
#         root_dir = os.path.split(input_dir)[0]
#         if not os.path.exists(os.path.join(root_dir, 'service_id.yml')):
#             raise ValueError(f'`--root-dir` is required but not specified.')
#
#     # open ID manager
#     id_manager = TraceGraphIDManager(root_dir)
#
#     return input_dirs, id_manager


@main.command()
@click.option('-i', '--input-dir')
@click.option('-r', '--root-dir', required=False, default=None)
@click.option('--names', required=False, default=None, multiple=True)
@click.option('--gui', is_flag=True, default=False)
@click.option('--node-count-hist-out', required=False, default=None)
@click.option('--max-depth-hist-out', required=False, default=None)
@click.option('--root-latency-hist-out', required=False, default=None)
def data_hist(input_dir, root_dir, names, gui,
              node_count_hist_out, max_depth_hist_out, root_latency_hist_out):
    db, id_manager = open_trace_graph_db(input_dir, root_dir, names)
    print('DB:', db)

    # load information
    rows = []
    for g in tqdm(db, desc=f'Process DB'):
        max_span_count = 0
        for _, nd in g.iter_bfs():
            max_span_count = max(max_span_count, nd.features.span_count)
        rows.append({
            'node_count': g.node_count,
            'max_depth': g.max_depth,
            'root_latency': g.root.features.avg_latency,
            'max_span_count': max_span_count,
        })
    df = pd.DataFrame(rows)

    # print stats
    q = [.9, .95, .99]
    q_df = df.quantile(q)
    stats = [
        {'key': 'count(graph)', 'value': len(rows)},
        {'key': 'max(graph.node_count)', 'value': df.node_count.max()},
        {'key': 'max(graph.max_depth)', 'value': df.max_depth.max()},
        {'key': 'max(graph.root_latency)', 'value': df.root_latency.max()},
        {'key': 'max(graph.max_span_count)', 'value': df.max_span_count.max()},
        {'key': f'%(graph.node_count, {q})', 'value': np.array(q_df.node_count).tolist()},
        {'key': f'%(graph.max_depth, {q})', 'value': np.array(q_df.max_depth).tolist()},
        {'key': f'%(graph.root_latency, {q})', 'value': np.array(q_df.root_latency).tolist()},
        {'key': f'%(graph.max_span_count, {q})', 'value': np.array(q_df.max_span_count).tolist()},
    ]
    print(pd.DataFrame(stats).set_index('key', drop=True))

    # plot discrete histogram
    def plot_hist(series, out_file):
        k = series.max()
        bins = np.linspace(0, k, k + 1, dtype=np.int32)
        sns.histplot(series, bins=bins, discrete=True)
        plt.xticks(bins)
        if gui:
            plt.show()
        elif out_file:
            plt.savefig(out_file)
        plt.close()

    if gui or node_count_hist_out:
        plot_hist(df.node_count, node_count_hist_out)
    if gui or max_depth_hist_out:
        plot_hist(df.max_depth, max_depth_hist_out)

    # plot root_latency histogram
    if gui or root_latency_hist_out:
        sns.histplot(df.root_latency, kde=False)
        if gui:
            plt.show()
        elif root_latency_hist_out:
            plt.savefig(root_latency_hist_out)
        plt.close()


@main.command()
@click.option('-i', '--input-dir')
@click.option('-r', '--root-dir', required=False, default=None)
@click.option('--names', required=False, default=None, multiple=True)
@click.option('--gui', is_flag=True, default=False)
@click.option('-o', '--output-file', default=None, required=False)
def latency_hist(input_dir, root_dir, names, gui, output_file):
    db, id_manager = open_trace_graph_db(input_dir, root_dir, names)
    print('DB:', db)

    # load information
    service_latencies = {i: [] for i in range(id_manager.num_operations)}
    for g in tqdm(db, desc=f'Process DB'):
        for _, nd in g.iter_bfs():
            assert isinstance(nd, TraceGraphNode)
            service_latencies[nd.operation_id].append(nd.features.avg_latency)

    service_latencies = {k: v for k, v in service_latencies.items() if v}

    # plot
    fig = plot_latency_hist(service_latencies, id_manager)
    plt.tight_layout()
    if gui:
        plt.show()
    elif output_file:
        plt.savefig(output_file)
    plt.close()


if __name__ == '__main__':
    main()
