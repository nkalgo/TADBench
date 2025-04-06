import warnings
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
import pandas as pd

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
@click.option('-n', '--name', type=str, required=True)
def csv_to_db(input_dir, output_dir, name):
    # check the parameters
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    input_path = os.path.join(input_dir, f"{name}.csv")
    output_path = os.path.join(output_dir, "processed", name)

    # Load id_manager
    id_manager = TraceGraphIDManager(os.path.join(input_dir, 'id_manager'))

    # process the traces
    # load the graphs
    if 'test' not in name:
        df = load_trace_csv(input_path)
        trace_graphs = df_to_trace_graphs(
            df,
            id_manager=id_manager,
            merge_spans=True,
        )

        # write to db
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        db = BytesSqliteDB(output_path, write=True)
        with db, db.write_batch():
            for g in tqdm(trace_graphs, desc='Save graphs'):
                db.add(g.to_bytes())
    else:
        # read test data
        df = load_trace_csv(input_path, is_test=True)

        for i in range(4):
            trace_graphs = df_to_trace_graphs(
                df,
                id_manager=id_manager,
                merge_spans=True,
                test_label=i
            )

            # write to db
            if i == 0:
                output_path = os.path.join(output_dir, 'processed', 'test')
            elif i == 1:
                output_path = os.path.join(output_dir, 'processed', 'test-drop')
            elif i == 2:
                output_path = os.path.join(output_dir, 'processed', 'test-latency')
            elif i == 3:
                output_path = os.path.join(output_dir, 'processed', 'test-both')

            if os.path.exists(output_path):
                shutil.rmtree(output_path)

            db = BytesSqliteDB(output_path, write=True)
            with db, db.write_batch():
                for g in tqdm(trace_graphs, desc='Save graphs'):
                    db.add(g.to_bytes())



@main.command()
@click.option('-i', '--input-dir')
@click.option('-F', '--force-regenerate', is_flag=True, default=True)
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
            f[k] = {
                'mean': np.mean(v) if len(v) > 1 else 0,
                'std': np.std(v) if len(v) > 1 else 0,
                'p99': np.percentile(v, 99) if len(v) > 1 else 0,
            }


@main.command()
@click.option('-i', '--input-file')
@click.option('-o', '--output-dir')
def make_status_id(input_file, output_dir):
    df = pd.read_csv(input_file)
    status_set = set(df['status'].values.tolist())
    with open(os.path.join(output_dir, "status_id.yml"), 'w') as f:
        f.write("? ''\n")
        f.write(": 0\n")
        for i,status in enumerate(status_set):
            f.write(f"'{status}': {i+1}\n")


@main.command()
@click.option('-i', '--input-dir')
@click.option('-o', '--output-dir')
def make_status_id_train_and_val_and_test(input_dir, output_dir):
    df1 = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    status_set_train = set(df1['status'].values.tolist())
    df2 = pd.read_csv(os.path.join(input_dir, 'val.csv'))
    status_set_val = set(df2['status'].values.tolist())
    df3 = pd.read_csv(os.path.join(input_dir, 'test.csv'))
    status_set_test = set(df3['status'].values.tolist())
    status_set = status_set_train | status_set_val | status_set_test
    with open(os.path.join(output_dir, "status_id.yml"), 'w') as f:
        f.write("? ''\n")
        f.write(": 0\n")
        for i, status in enumerate(status_set):
            f.write(f"'{status}': {i + 1}\n")


@main.command()
@click.option('-i', '--input-file')
@click.option('-o', '--output-dir')
def make_operation_id(input_file, output_dir):
    df = pd.read_csv(input_file)
    service_set = set(df['serviceName'].values.tolist())
    operation_set = set(df['operationName'].values.tolist())
    res_com = []
    for service in service_set:
        for operation in operation_set:
            res_com.append(f"{service}/{operation}")
    with open(os.path.join(output_dir, "operation_id.yml"),'w') as f:
        f.write("? ''\n")
        f.write(": 0\n")
        for i,j in enumerate(res_com):
            f.write(f"'{j}': {i+1}\n")



@main.command()
@click.option('-i', '--input-dir')
@click.option('-o', '--output-dir')
def make_operation_id_train_and_val_and_test(input_dir, output_dir):
    df1 = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    service_set_train = set(df1['serviceName'].values.tolist())
    operation_set_train = set(df1['operationName'].values.tolist())

    df2 = pd.read_csv(os.path.join(input_dir, 'val.csv'))
    service_set_val = set(df2['serviceName'].values.tolist())
    operation_set_val = set(df2['operationName'].values.tolist())

    df3 = pd.read_csv(os.path.join(input_dir, 'test.csv'))
    service_set_test = set(df3['serviceName'].values.tolist())
    operation_set_test = set(df3['operationName'].values.tolist())

    service_set = service_set_train | service_set_val | service_set_test
    operation_set = operation_set_train | operation_set_val | operation_set_test

    res_com = []
    for service in service_set:
        for operation in operation_set:
            res_com.append(f"{service}/{operation}")
    with open(os.path.join(output_dir, "operation_id.yml"), 'w') as f:
        f.write("? ''\n")
        f.write(": 0\n")
        for i, j in enumerate(res_com):
            f.write(f"'{j}': {i + 1}\n")


@main.command()
@click.option('-i', '--input-file')
@click.option('-o', '--output-dir')
def make_service_id(input_file, output_dir):
    df = pd.read_csv(input_file)
    service_set = set(df['serviceName'].values.tolist())
    with open(os.path.join(output_dir, "service_id.yml"), 'w') as f:
        f.write("? ''\n")
        f.write(": 0\n")
        for i, service in enumerate(service_set):
            f.write(f"'{service}': {i + 1}\n")



@main.command()
@click.option('-i', '--input-dir')
@click.option('-o', '--output-dir')
def make_service_id_train_and_val_and_test(input_dir, output_dir):
    df1 = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    service_set_train = set(df1['serviceName'].values.tolist())

    df2 = pd.read_csv(os.path.join(input_dir, 'val.csv'))
    service_set_val = set(df2['serviceName'].values.tolist())

    df3 = pd.read_csv(os.path.join(input_dir, 'test.csv'))
    service_set_test = set(df3['serviceName'].values.tolist())

    service_set = service_set_train | service_set_val | service_set_test
    with open(os.path.join(output_dir, "service_id.yml"), 'w') as f:
        f.write("? ''\n")
        f.write(": 0\n")
        for i, service in enumerate(service_set):
            f.write(f"'{service}': {i + 1}\n")


@main.command()
@click.option('-i', '--input-dir')
@click.option('-o', '--output_dir')
def preprocess(input_dir, output_dir):
    print("Convert datasets...")
    print("------------> Train")
    os.system(f"python3 -m tracegnn.cli.data_process csv-to-db -i {input_dir} -o {output_dir} -n train")
    print("------------> Val")
    os.system(f"python3 -m tracegnn.cli.data_process csv-to-db -i {input_dir} -o {output_dir} -n val")
    print("------------> Test")
    os.system(f"python3 -m tracegnn.cli.data_process csv-to-db -i {input_dir} -o {output_dir} -n test")

    print("Finished!")


if __name__=='__main__':
    main()
