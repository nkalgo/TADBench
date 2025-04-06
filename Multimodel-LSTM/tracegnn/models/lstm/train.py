import os
import shutil
from typing import *

from tracegnn.models.fsa.utils import Node, Trace
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tracegnn.data import *
from .utils import analyze_multiclass_anomaly_nll
from .model import calculate_score

from tqdm import tqdm
import numpy as np
import click



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
            trace = {}
            trace['anomaly'] = 0 if not g.data.get('is_anomaly') else (
                1 if g.data['anomaly_type'] == 'drop' else (
                    2 if g.data['anomaly_type'] == 'latency' else 3))
            # Change '-' to 0 in trace_id to avoid bugs
            # trace['trace_id'] = (str(g.trace_id[0]) + str(g.trace_id[1])).replace('-', '0')
            trace['trace_id'] = g.trace_id
            trace['labels'] = []
            trace['latency'] = []
            trace['node_anomaly'] = []
            
            # DFS
            stack = []
            stack.append((None, g.root))

            while stack:
                parent, node = stack.pop()
                if parent is not None:
                    trace['labels'].append(f'{parent.operation_id}#{node.operation_id}')
                    trace['latency'].append(int(node.features.avg_latency))
                    trace['node_anomaly'].append(node.anomaly)
                else:
                    trace['labels'].append(f'{node.operation_id}')
                    trace['latency'].append(int(node.features.avg_latency))
                    trace['node_anomaly'].append(node.anomaly)


                for child in sorted(node.children, key=lambda x: x.operation_id, reverse=True):
                    stack.append((node, child))

            # Return result
            result.append(trace)

    return result

@click.command()
@click.option('--dataset', required=True, type=str)
@click.option('--device', required=False, type=str, default='cuda:1')
@click.option('--nt', required=False, type=int, default=20)
@click.option('--data3', is_flag=True, default=False, required=False)
@click.option('--data4', is_flag=True, default=False, required=False)
@click.option('--drop2', is_flag=True, default=False, required=False)
@click.option('--ltest', is_flag=True, default=False, required=False)
def main(dataset, device, nt, data3, data4, drop2, ltest):
    import time
    start_time = time.time()
    print('LSTM: Loading data...')

    # Load data
    dataset_path = f'./Datasets/{dataset}/after_process_db'
    # shutil.copytree(f'../TraceVAE/data/GAIA/processed', dataset_path)
    # shutil.copytree(f'TraceVAE/data/GAIA/id_manager', dataset_path)
    print(os.listdir(dataset_path))
    # dataset_path = f'/srv/data/tracegnn/{dataset}/processed'
    train_db = make_bytes_db(dataset_path, ['train'])
    test_db = make_bytes_db(dataset_path, ['test', 'test-drop', 'test-latency', 'test-both'])
    # test_db = make_bytes_db(dataset_path, ['test', 'ltest-drop', 'ltest-latency'])

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

    # test_db = make_bytes_db(dataset_path, data_names)
    # test_db = train_db
    

    train_traces = get_trace_dict(train_db)
    '''
    trace = {
               'trace_id': 'ddd',
               'anomaly':0/1/2/3,
               'labels': ['{parent.operation_id}#{node.operation_id}', ...]
               'latency': [int(node.features.avg_latency), ...]
            }
    '''
    test_traces = get_trace_dict(test_db)
    node_labels = []
    for i in test_traces:
        for j in i['node_anomaly']:
            node_labels.append(j)
    # print(node_labels)
    # test_traces = train_traces
    # Get result
    print('LSTM: Calculating...')
    nll_latency, nll_drop, labels, node_latency, node_labels, test_time = calculate_score(
        train_traces=train_traces,
        test_traces=test_traces,
        model_label=model_label,
        dataset=dataset,
        Nt=nt,
        device=device
    )
    # print('node_labels', node_labels)
    print('all_train_time', time.time()-start_time)
    print('test time', test_time)
    start_time = time.time()
    print(analyze_multiclass_anomaly_nll(
        nll_latency=np.array(nll_latency, dtype=np.float32),
        nll_drop=np.array(nll_drop, dtype=np.float32),
        label_list=np.array(labels, dtype=np.int32),
        node_latency=np.array(node_latency, dtype=np.float32),
        node_label_list=np.array(node_labels, dtype=np.int32),
        method='lstm',
        dataset=dataset,
        save_dict=True,
        save_filename=save_filename
    ))
    print("test+calculate time:",test_time + time.time()-start_time,'s')


if __name__ == '__main__':
    main()
