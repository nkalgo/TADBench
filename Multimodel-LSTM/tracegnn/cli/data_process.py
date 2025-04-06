import math
import pickle as pkl
import random
import shutil
import sys

import click
import numpy as np
from tqdm import tqdm

from tracegnn.constants import *
from tracegnn.data import *
from tracegnn.utils import *


def get_graph_key(g):
    node_types = set()
    stack = [g.root]
    while stack:
        nd = stack.pop()
        node_types.add(nd.operation_id)
        stack.extend(nd.children)
    return g.root.operation_id, g.max_depth, tuple(sorted(node_types))


@click.group()
def main():
    pass


@main.command()
@click.option('-i', '--input-dir')
@click.option('-o', '--output-dir')
@click.option('--one-pass', is_flag=True, default=False)
@click.option('-M', '--merge-spans', is_flag=True, default=False)
@click.option('-F', '--force-regenerate', is_flag=True, default=False)
def csv_to_db(input_dir, output_dir, one_pass, merge_spans, force_regenerate):
    # check the parameters
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    # gather input files
    jobs = []
    for name in os.listdir(input_dir):
        for ext in ('.csv', '.csv.xz', '.csv.gz'):
            if name.endswith(ext):
                input_path = os.path.join(input_dir, name)
                output_path = os.path.join(output_dir, name[:-len(ext)])
                if force_regenerate or not (os.path.exists(output_path) and os.listdir(output_path)):
                    jobs.append((input_path, output_path))

    # prepare for the working directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    id_manager = TraceGraphIDManager(output_dir)

    # preload the files
    loaded_files = {}
    service_names = set()
    operation_names = set()

    if not one_pass:
        for input_path, output_path in tqdm(jobs, desc='Load files'):
            df = loaded_files[input_path] = load_trace_csv(input_path)
            for row in tqdm(df.itertuples(), desc=f'Scan service and operations from {input_path}', total=len(df)):
                if row.DBhash == 0:
                    service_names.add(row.serviceName)
                    operation_names.add(f'{row.serviceName}/{row.operationName}')
        with id_manager:
            for service_name in sorted(service_names):
                id_manager.service_id.get_or_assign(service_name)
            for operation_name in sorted(operation_names):
                id_manager.operation_id.get_or_assign(operation_name)

    # process the traces
    for input_path, output_path in tqdm(jobs, desc='Process files'):
        # load the graphs
        df = loaded_files.get(input_path)
        if df is None:
            df = load_trace_csv(input_path)
        trace_graphs = df_to_trace_graphs(
            df,
            id_manager=id_manager,
            merge_spans=merge_spans,
        )

        # write to db
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        db = BytesSqliteDB(output_path, write=True)
        with db, db.write_batch():
            for g in tqdm(trace_graphs, desc='Save graphs'):
                db.add(g.to_bytes())


@main.command()
@click.option('-i', '--input-dir')
@click.option('-o', '--output-dir')
@click.option('--train-size', type=int, required=True)
@click.option('--val-size', type=int, required=True)
@click.option('--test-size', type=int, required=True)
@click.option('-M', '--merge-spans', is_flag=True, default=False)
@click.option('-F', '--force-regenerate', is_flag=True, default=False)
@click.option('--min-bin-size', type=int, required=True, default=100)
@click.option('--max-root-latency', type=float, required=True, default=5000)
@click.option('--max-node-count', type=int, required=True, default=MAX_NODE_COUNT)
@click.option('--max-depth', type=int, required=True, default=MAX_DEPTH)
def make_train_test(input_dir, output_dir, train_size, val_size, test_size,
                    merge_spans, force_regenerate,
                    min_bin_size, max_root_latency, max_node_count, max_depth):
    def pr(*args, file=sys.stderr):
        sys.stderr.flush()
        sys.stdout.flush()
        print(*args, file=file)
        sys.stderr.flush()
        sys.stdout.flush()

    # open input db
    db, id_manager = open_trace_graph_db(input_dir)

    # check the output dir
    output_dir = os.path.abspath(output_dir)
    if os.path.isdir(output_dir):
        if not force_regenerate:
            print(f'`output_dir` already exists: {output_dir}', file=sys.stderr)
            exit(0)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # copy id manager to output
    id_manager.dump_to(output_dir)

    # load the graphs and put into bins
    graph_bins = {}
    total_count = 0
    too_large_node_count = 0
    too_large_root_latency = 0
    too_large_depth = 0

    for g in tqdm(db, desc=f'Load from {input_dir}'):
        total_count += 1
        if merge_spans:
            g.merge_spans_and_assign_id()
        if g.node_count > max_node_count:
            too_large_node_count += 1
            continue
        if g.root.features.avg_latency > max_root_latency:
            too_large_root_latency += 1
            continue
        if g.max_depth > max_depth:
            too_large_depth += 1
            continue
        g_key = get_graph_key(g)

        if g_key not in graph_bins:
            graph_bins[g_key] = []
        graph_bins[g_key].append(g)

    if too_large_node_count or too_large_root_latency or too_large_depth:
        pr(
            f'Dropped graph stats:\n'
            f'  {too_large_node_count} ({too_large_node_count / total_count * 100.:.2f}%) too large node_count\n'
            f'  {too_large_root_latency} ({too_large_root_latency / total_count * 100.:.2f}%) too large root_latency\n'
            f'  {too_large_depth} ({too_large_depth / total_count * 100.:.2f}%) too large max_depth\n',
            file=sys.stdout
        )

    if False:  # inspect the cdf of root_latency
        keys, values = compute_cdf(
            np.array(
                sum(
                    [
                        [g.root.features.avg_latency for g in v]
                        for v in graph_bins.values()
                    ],
                    []
                ),
                dtype=np.float64
            )
        )
        offset = len(values)
        offset = np.where(values >= 0.99)[0][0]

        from matplotlib import pyplot as plt
        plt.plot(keys[:offset], values[:offset])
        plt.show()
        exit(0)

    if False:  # inspect the cdf of bin_size
        keys, values = compute_cdf(np.array([len(v) for v in graph_bins.values()], dtype=np.float64))
        offset = np.where(values >= 0.99)[0][0]

        from matplotlib import pyplot as plt
        plt.plot(keys[:offset], values[:offset])
        plt.show()
        exit(0)

    # filter the graph_bins according to the min_bin_size
    old_graph_count = sum(map(len, graph_bins.values()))
    old_bin_count = len(graph_bins)

    graph_bins = {k: v for k, v in graph_bins.items() if len(v) >= min_bin_size}

    new_graph_count = sum(map(len, graph_bins.values()))
    new_bin_count = len(graph_bins)

    drop_graph_count = old_graph_count - new_graph_count
    drop_bin_count = old_bin_count - new_bin_count
    pr(
        f'Dropped {drop_graph_count} ({drop_graph_count / old_graph_count * 100.:.2f}%) graphs, '
        f'which are from the smallest {drop_bin_count} ({drop_bin_count / old_bin_count * 100.:.2f}%) bins.',
        file=sys.stdout
    )

    # split train and test
    train_graph_bins = {}
    val_graph_bins = {}
    test_graph_bins = {}
    val_ratio = val_size / (train_size + val_size + test_size)
    test_ratio = test_size / (train_size + val_size + test_size)

    for key in graph_bins:
        bin = np.array(graph_bins[key])
        bin_size = len(bin)
        bin_val_size = int(math.ceil(bin_size * val_ratio))
        bin_test_size = int(math.ceil(bin_size * test_ratio))
        bin_train_size = bin_size - bin_val_size - bin_test_size

        train_graph_bins[key] = bin[: bin_train_size]
        val_graph_bins[key] = bin[bin_train_size: bin_train_size + bin_val_size]
        test_graph_bins[key] = bin[bin_train_size + bin_val_size:]

    # do resample
    def do_resample(graph_bins, output_path, population_size):
        bin_count = len(graph_bins)
        bin_sample_size = (population_size + bin_count - 1) // bin_count
        trace_graphs = []
        desc = f'Sample graphs ({bin_sample_size} * {bin_count} bins) for {output_path}'
        pr(desc, file=sys.stdout)

        for bin_array in tqdm(graph_bins.values(), desc=desc):
            if len(bin_array) < bin_sample_size:
                # up sampling
                for i in np.random.randint(len(bin_array), size=bin_sample_size):
                    trace_graphs.append(bin_array[i])
            else:
                # down sampling
                random.shuffle(bin_array)
                trace_graphs.extend(bin_array[:bin_sample_size])

        # write to db
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        db = BytesSqliteDB(output_path, write=True)
        with db, db.write_batch():
            for g in tqdm(trace_graphs, desc=f'Save {len(trace_graphs)} resampled graphs to {output_path}'):
                db.add(g.to_bytes())

    do_resample(train_graph_bins, os.path.join(output_dir, 'train'), train_size)
    do_resample(val_graph_bins, os.path.join(output_dir, 'val'), val_size)
    do_resample(test_graph_bins, os.path.join(output_dir, 'test'), test_size)


@main.command()
@click.option('-i', '--input-dir')
@click.option('-F', '--force-regenerate', is_flag=True, default=False)
@click.option('--names', multiple=True, required=False, default=None)
def make_latency_range(input_dir, force_regenerate, names):
    db, id_manager = open_trace_graph_db(input_dir, names=names)

    f = TraceGraphLatencyRangeFile(id_manager.root_dir)
    if os.path.exists(f.yaml_path) and not force_regenerate:
        print(f'LatencyRangeFile already exists: {f.yaml_path}', file=sys.stderr)
        exit(0)

    latency_map = {
        i: []
        for i in range(id_manager.num_operations)
    }
    for g in tqdm(db, desc='Process graphs'):
        for _, nd in g.iter_bfs():
            assert isinstance(nd, TraceGraphNode)
            latency_map[nd.operation_id].append(nd.features.max_latency)

    with f:
        f.clear()
        for k, v in latency_map.items():
            if len(v) > 1:
                f[k] = {
                    'mean': np.mean(v),
                    'std': np.std(v),
                    'p99': np.percentile(v, 99),
                }


@main.command()
@click.option('-i', '--input-dir', required=True)
@click.option('-o', '--output-dir', required=True)
@click.option('-O', '--origin-output-dir', required=False, default=None)
@click.option('-N', '--test-size', type=int, required=False, default=False)
@click.option('--one-to-one', is_flag=True, required=False, default=False)
@click.option('--just-one-node', is_flag=True, required=False, default=False)
@click.option('--avg-as-whole', is_flag=True, required=False, default=False)
@click.option('--set-as-normal', is_flag=True, required=False, default=False)
@click.option('--drop-ratio', type=float, nargs=2, required=False, default=None)
@click.option('--drop-subtract-latency', is_flag=True, required=False, default=False)
@click.option('--latency-ratio', type=float, nargs=2, required=False, default=None)
@click.option('--latency-delta', type=float, nargs=2, required=False, default=None)
@click.option('--latency-delta-nstd', type=float, nargs=2, required=False, default=None)
@click.option('--latency-delta-np99', type=float, nargs=2, required=False, default=None)
@click.option('--latency-p99-min', type=float, required=False, default=None)
@click.option('-F', '--force-regenerate', is_flag=True, default=False)
def make_synthetic_test(input_dir, output_dir, origin_output_dir, test_size, one_to_one,
                        just_one_node, avg_as_whole, set_as_normal,
                        drop_ratio, drop_subtract_latency, latency_ratio,
                        latency_delta, latency_delta_nstd, latency_delta_np99, latency_p99_min,
                        force_regenerate):
    # check parameters
    if test_size is None and not one_to_one:
        raise ValueError(f'Either of `--test-size` and `--one-to-one` should be specified.')
    if not drop_ratio and not latency_ratio:
        raise ValueError('`--drop-ratio` and `--latency-ratio` cannot be both zero.')
    if latency_ratio and ((not latency_delta) + (not latency_delta_nstd) + (not latency_delta_np99) != 2):
        raise ValueError('One and only one of `--latency-delta`, `--latency-delta-nstd` '
                         'and `--latency_delta_np99` should be specified.')

    # open input db
    db, id_manager = open_trace_graph_db(input_dir)
    latency_range = TraceGraphLatencyRangeFile(id_manager.root_dir, require_exists=True)

    # check the output dir and origin_output_dir
    output_dir = os.path.abspath(output_dir)
    paths = [output_dir]

    if origin_output_dir is not None:
        origin_output_dir = os.path.abspath(origin_output_dir)
        paths.append(origin_output_dir)

    for path in paths:
        if os.path.isdir(path):
            if not force_regenerate:
                print(f'Path already exists: {path}', file=sys.stderr)
                exit(0)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    # load the graphs
    old_graphs = []
    for g in tqdm(db, desc=f'Load from {input_dir}'):
        old_graphs.append(g)

    # inspect the graph keys
    graph_keys = {get_graph_key(g) for g in old_graphs}
    print(f'Graph key count: {len(graph_keys)}')

    # now do mutate
    def make_mutate_graph():
        if one_to_one:
            def iter_graphs():
                return old_graphs
        else:
            def iter_graphs():
                while count[0] < test_size:
                    yield old_graphs[np.random.randint(len(old_graphs))]

        count = [0]
        for g in iter_graphs():
            kw = {
                'just_one_node': just_one_node,
                'avg_as_whole': avg_as_whole,
                'drop_ratio': drop_ratio,
                'drop_subtract_latency': drop_subtract_latency,
                'latency_ratio': latency_ratio,
                'latency_delta': latency_delta,
                'latency_delta_nstd': latency_delta_nstd,
                'latency_delta_np99': latency_delta_np99,
                'latency_p99_min': latency_p99_min,
            }

            # choose one way to mutate
            if drop_ratio and latency_ratio:
                if np.random.binomial(n=1, p=.5) == 0:
                    kw['drop_ratio'] = None
                else:
                    kw['latency_ratio'] = None

            # mark the anomaly type
            anomaly_type = 'drop' if kw['drop_ratio'] else 'latency'

            # mutate the graph
            g2_okay = False

            while not g2_okay:
                g2 = trace_graph_mutate(g, latency_range, **kw)
                if (g2 is not g) and (g2.node_count >= 1):
                    g2.data['anomaly_type'] = anomaly_type
                    g2_okay = True
                    if anomaly_type == 'drop':
                        if get_graph_key(g2) in graph_keys:
                            g2_okay = False  # maybe a valid graph, thus we need to throw away it

                if g2_okay:
                    count[0] += 1
                    yield g, g2
                    break

                if not one_to_one:
                    # throw `g` and use the next input `g`
                    break

    origin_graphs = []
    new_graphs = []

    for origin_g, new_g in tqdm(make_mutate_graph(),
                                desc='Mutate graphs',
                                total=test_size):
        if origin_output_dir:
            origin_graphs.append(origin_g)
        if set_as_normal:
            new_g.data['is_anomaly'] = False
            new_g.data['anomaly_type'] = 0
        new_graphs.append(new_g)

    # save the graphs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    db = BytesSqliteDB(output_dir, write=True)

    if origin_output_dir:
        if os.path.exists(origin_output_dir):
            shutil.rmtree(origin_output_dir)
        origin_db = BytesSqliteDB(origin_output_dir, write=True)

        desc = f'Save {len(new_graphs)} mutated graphs to {output_dir}, and ' \
               f'{len(origin_graphs)} origin graphs to {origin_output_dir}'
        print(desc)
        with db, origin_db, db.write_batch(), origin_db.write_batch():
            for origin_g, new_g in tqdm(zip(origin_graphs, new_graphs),
                                        desc=desc,
                                        total=len(origin_graphs)):
                origin_db.add(origin_g.to_bytes())
                db.add(new_g.to_bytes())

    else:
        desc = f'Save {len(new_graphs)} mutated graphs to {output_dir}'
        print(desc)
        with db, db.write_batch():
            for g in tqdm(new_graphs, desc=desc):
                db.add(g.to_bytes())


@main.command()
@click.option('-i', '--input-dir')
@click.option('-p', '--protocol', required=True, type=int, default=pkl.DEFAULT_PROTOCOL)
@click.option('--names', multiple=True, required=False, default=None)
def downgrade_protocol(input_dir, protocol, names):
    if protocol >= pkl.HIGHEST_PROTOCOL:
        raise ValueError(f'No need to downgrade.')

    if not names:
        names = []
        for name in os.listdir(input_dir):
            path = os.path.join(input_dir, name, '_bytes.db')
            if os.path.isfile(path):
                names.append(name)

    for name in tqdm(names):
        db_path = os.path.join(input_dir, name)

        # copy from one db to another
        src_db, id_manager = open_trace_graph_db(db_path)
        dst_db = TraceGraphDB(
            BytesSqliteDB(
                db_path,
                file_name=f'_bytes_{protocol}.db',
                write=True,
            ),
            protocol=protocol,
        )
        with src_db, dst_db:
            for g in tqdm(src_db, desc=f'Process {name}'):
                dst_db.add(g)

        # now replace the original db
        src_path = os.path.join(db_path, f'_bytes_{protocol}.db')
        dst_path = os.path.join(db_path, f'_bytes.db')
        tmp_path = os.path.join(db_path, f'_bytes.db.backup')
        try:
            os.rename(dst_path, tmp_path)
            os.rename(src_path, dst_path)
        finally:
            if not os.path.exists(dst_path):
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, dst_path)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    main()
