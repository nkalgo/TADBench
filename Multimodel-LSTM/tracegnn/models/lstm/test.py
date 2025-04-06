import os
from typing import *

import math

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tracegnn.data import *
from .utils import analyze_multiclass_anomaly_nll
from .model import calculate_score

from tqdm import tqdm
import numpy as np
import click


class Trace:
    def __init__(self, flowid, msgs):
        self.f_id = flowid
        self.s = msgs
        self.n = len(msgs)
        for i in range(0, self.n):
            self.s[i] = int(self.s[i][0]), int(self.s[i][1])

    def key(self, k):
        if k >= self.n or k < 0:
            return -1
        return self.s[k][0]

    def time(self, k):
        if k >= self.n or k < 0:
            return -1
        return self.s[k][1]

    def is_empty(self):
        return self.n == 0


class Node:
    def __init__(self, id):
        self.acc = False
        self.edges = {}
        self.id = id
        self.ab = {}
        self.sum = {}
        self.sum2 = {}
        self.num = {}

    def equal(self, other):
        return self.id == other.id

    def add_edge(self, x, y):
        if self.edges.__contains__(x):
            if y.equal(self.edges[x]):
                return True
            return False
        else:
            self.edges[x] = y
            return True

    def add_time(self, x, y):
        if self.sum.__contains__(x):
            self.sum[x] = self.sum[x] + y
            self.sum2[x] = self.sum2[x] + y * y
            self.num[x] = self.num[x] + 1
        else:
            self.sum[x] = y
            self.sum2[x] = y * y
            self.num[x] = 1

    def go(self, x):
        if self.edges.__contains__(x):
            return self.edges[x]
        else:
            return None

    def set_final(self, flag):
        self.acc = flag

    def get_ab(self):
        for key in self.edges:
            self.ab[key] = self.iterate(self.num[key],
                                        self.sum[key],
                                        self.sum2[key])

    def iterate(self, k, x, y):
        # k: number
        # x: sum
        # y: sqr sum

        # initialization:
        a = 1.0 * x / k
        b = b0 = 0.0

        # iteration
        while True:
            b0 = b
            b = k * a * a + y - 2 * a * x
            b = 1.0 * b / (k * a * a)
            if math.fabs(b - b0) < 1e-5:
                break
            else:
                a = math.sqrt(x * x + 4 * k * b * y) - x
                a = 1.0 * a / (2 * k * b)

        return a * (1 + 3 * math.sqrt(b))


def make_bytes_db(input_dir: str, names: List[str]) -> BytesDB:
    if len(names) > 1:
        return BytesMultiDB(*[
            BytesSqliteDB(os.path.join(input_dir, name))
            for name in names
        ])
    return BytesSqliteDB(os.path.join(input_dir, names[0]))


def get_trace_dict(bytes_db: BytesDB) -> List[dict]:
    result: List[dict] = []
    with TraceGraphDB(bytes_db) as db:
        g: TraceGraph
        for g in tqdm(db, desc=f'Process...'):
            # print(g.trace_id)
            trace = {}
            trace['anomaly'] = 0 if not g.data.get('is_anomaly') else (
                1 if g.data['anomaly_type'] == 'drop' else (
                    2 if g.data['anomaly_type'] == 'latency' else 3))
            # Change '-' to 0 in trace_id to avoid bugs
            trace['trace_id'] = (str(g.trace_id[0]) + str(g.trace_id[1])).replace('-', '0')
            trace['trace_id'] = g.trace_id
            trace['labels'] = []
            trace['latency'] = []
            
            # DFS
            stack = []
            stack.append((None, g.root))

            while stack:
                parent, node = stack.pop()
                if parent is not None:
                    trace['labels'].append(f'{parent.operation_id}#{node.operation_id}')
                    trace['latency'].append(int(node.features.avg_latency))
                else:
                    trace['labels'].append(f'{node.operation_id}')
                    trace['latency'].append(int(node.features.avg_latency))

                for child in sorted(node.children, key=lambda x: x.operation_id, reverse=True):
                    stack.append((node, child))

            # Return result
            result.append(trace)

    return result


@click.command()
@click.option('--dataset', required=True, type=str)
@click.option('--device', required=False, type=str, default='cuda')
@click.option('--nt', required=False, type=int, default=20)
@click.option('--data3', is_flag=True, default=False, required=False)
@click.option('--data4', is_flag=True, default=False, required=False)
@click.option('--drop2', is_flag=True, default=False, required=False)
@click.option('--ltest', is_flag=True, default=False, required=False)
@click.option('--no-biased', is_flag=True, default=False, required=False)
def main(dataset, device, nt, data3, data4, drop2, ltest, no_biased):
    print('LSTM: Loading data...')

    # Load data
    dataset_path = f'./Datasets/{dataset}/after_process_db'
    train_db = make_bytes_db(dataset_path, ['train'])
    test_db = make_bytes_db(dataset_path, ['test', 'test-drop', 'test-latency'])

    # load the dataset
    if ltest:
        data_names = ['test', 'ltest-drop', 'ltest-latency']
        save_filename = 'baselinel.csv'
        model_label = f'model_{dataset}_ltest.pkl'
    else:
        if drop2:
            data_names = ['test', 'test-drop-anomaly2', 'test-latency-anomaly2']
            save_filename = 'baseline2.csv'
            model_label = f'model_{dataset}_test2.pkl'
        elif data3:
            data_names = ['test', 'test-drop-anomaly3', 'test-latency-anomaly3']
            save_filename = 'baseline3.csv'
            model_label = f'model_{dataset}_test3.pkl'
        elif data4:
            data_names = ['test', 'test-drop-anomaly4', 'test-latency-anomaly4']
            save_filename = 'baseline4.csv'
            model_label = f'model_{dataset}_test4.pkl'
        else:
            # data_names = ['test', 'test-drop-anomaly', 'test-latency-anomaly2']
            data_names = ['train.csv']
            save_filename = f'{dataset}_result.csv'
            model_label = f'model_{dataset}_test.pkl'


    train_traces = get_trace_dict(train_db)
    test_traces = get_trace_dict(test_db)

    # Get result
    print('LSTM: Evaluation...')
    import time
    start_time = time.time()
    nll_latency, nll_drop, labels = calculate_score(
        train_traces=train_traces,
        test_traces=test_traces,
        dataset=dataset,
        Nt=nt,
        device=device,
        test_only=True,
        model_label=model_label,
        no_bias=no_biased
    )
    print(analyze_multiclass_anomaly_nll(
        nll_latency=np.array(nll_latency, dtype=np.float32),
        nll_drop=np.array(nll_drop, dtype=np.float32),
        label_list=np.array(labels, dtype=np.int32),
        method='lstm',
        dataset=dataset,
        save_dict=True,
        save_filename=save_filename
    ))
    print("test time:", time.time() - start_time, 's')

    with open(f"tracegnn/models/lstm/model_dat/test_time_{dataset}.txt",'w') as f:
        f.write("test time: {}s".format(time.time()-start_time))

    import pickle
    with open(f"tracegnn/models/lstm/model_dat/nll_latency_{dataset}.pkl",'bw') as f:
        # print(type(nll_latency)) #list
        pickle.dump(nll_latency,f)
    with open(f"tracegnn/models/lstm/model_dat/labels_{dataset}.pkl",'bw') as f:
        # print(type(labels)) #list
        pickle.dump(labels, f)
    

if __name__ == '__main__':
    main()
    