import os

import math
import pickle as pkl
import random
import shutil
import sys

import click
import numpy as np
from tqdm import tqdm

from tracegnn.data import *
from tracegnn.utils import *
import pandas as pd



@click.group()
def \
        main():
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
    else:
        df = load_trace_csv(input_path, is_test=True)

    trace_graphs = df_to_trace_graphs(
        df,
        id_manager,
        name,
        merge_spans=True
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
@click.option('-o', '--output_dir')
def preprocess(input_dir, output_dir):
    print("Convert datasets...")
    print("------------> Train")
    os.system(f"python3 -m tracegnn.data.data_process csv-to-db -i {input_dir} -o {output_dir} -n train")
    print("------------> Val")
    os.system(f"python3 -m tracegnn.data.data_process csv-to-db -i {input_dir} -o {output_dir} -n val")
    print("------------> Test")
    os.system(f"python3 -m tracegnn.data.data_process csv-to-db -i {input_dir} -o {output_dir} -n test")

    print("Finished!")

# @main.command()
# @click.option('-i', '--input-dir')
# @click.option('-F', '--force-regenerate', is_flag=True, default=False)
# @click.option('--names', multiple=True, required=False, default=None)
# def make_latency_range(input_dir, force_regenerate, names):
#     db, id_manager = open_trace_graph_db(input_dir, names=names)
#
#     f = TraceGraphLatencyRangeFile(id_manager.root_dir)
#     if os.path.exists(f.yaml_path) and not force_regenerate:
#         print(f'LatencyRangeFile already exists: {f.yaml_path}', file=sys.stderr)
#         exit(0)
#
#     latency_map = {
#         i: []
#         for i in range(id_manager.num_operations)
#     }
#     for g in tqdm(db, desc='Process graphs'):
#         for _, nd in g.iter_bfs():
#             assert isinstance(nd, TraceGraphNode)
#             latency_map[nd.operation_id].append(nd.features.max_latency)
#
#     with f:
#         f.clear()
#         for k, v in latency_map.items():
#             # if len(v) > 1:
#             f[k] = {
#                 'mean': np.mean(v),
#                 'std': np.std(v),
#                 'p99': np.percentile(v, 99) if len(v) > 1 else 0,
#             }


if __name__ == '__main__':
    main()
