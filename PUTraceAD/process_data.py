import copy
import json
import os
import pickle
import random
from datetime import datetime
from queue import Queue
from data_format import *


def datetime_to_timestamp(dt):
    # dt = datetime.datetime(2020, 4, 11, 0, 4, 20, 366000)
    timestamp_10 = dt.timestamp()
    timestamp_13 = int(timestamp_10 * 1000) + dt.microsecond // 1000
    return timestamp_13


def get_spans(trace: Trace):
    spans = []
    q = Queue()
    q.put(trace.root_span)
    while not q.empty():
        span = q.get()
        spans.append(span)
        for child in span.children_span_list:
            q.put(child)
    return spans


def convert_to_putracead_format(traces: List[Trace]) -> List[dict]:
    random.shuffle(traces)
    converted_traces = []

    for trace in traces:
        putrace = {
            'nodes': [],
            'edges': [[], []],
            'abnormal': 0,
            'rc': '',
            'trace_id': trace.trace_id
        }

        if trace.anomaly_type == 0:
            putrace['abnormal'] = 0
        else:
            putrace['abnormal'] = 1

        spans = get_spans(trace)
        span_map = {span.span_id: span for span in spans}

        edge_index = 0
        edge_mappings = []

        for span in spans:
            children_duration = sum(
                child.duration for child in span.children_span_list
            )
            work_duration = max(0, span.duration - children_duration)

            parent_span_id = span.parent_span_id or '-1'

            node = {
                'spanId': span.span_id,
                'parentSpanId': parent_span_id,
                'startTime': datetime_to_timestamp(span.start_time),
                'rawDuration': span.duration,
                'workDuration': work_duration,
                'service': span.service_name,
                'operation': span.operation_name or span.service_name,
                'statusCode': int(span.status_code) if span.status_code else 0,
                'abnormal': span.anomaly
            }
            putrace['nodes'].append(node)

            if parent_span_id != '-1' and parent_span_id in span_map:
                parent_idx = spans.index(span_map[parent_span_id])
                current_idx = spans.index(span)
                edge_mappings.append((parent_idx, current_idx, edge_index))
                edge_index += 1

        for src_idx, dst_idx, edge_id in edge_mappings:
            putrace['edges'][0].append(edge_id)
            putrace['edges'][1].append(dst_idx)

        converted_traces.append(putrace)

    return converted_traces


def write_json_to_file(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


def putracead_preprocess(dataset_name, dataset_dir):
    with open(os.path.join(dataset_dir, 'train_normal.pkl'), 'rb') as f:
        train_normal_traces = pickle.load(f)
    with open(os.path.join(dataset_dir, 'test_normal.pkl'), 'rb') as f:
        test_normal_traces = pickle.load(f)
    with open(os.path.join(dataset_dir, 'abnormal.pkl'), 'rb') as f:
        test_abnormal_traces = pickle.load(f)
    with open(os.path.join(dataset_dir, 'only_latency_abnormal.pkl'), 'rb') as f:
        test_only_latency_abnormal_traces = pickle.load(f)
    with open(os.path.join(dataset_dir, 'only_structure_abnormal.pkl'), 'rb') as f:
        test_only_structure_abnormal_traces = pickle.load(f)
    print('converting to putracead format...')
    train_normal_data = convert_to_putracead_format(train_normal_traces)
    test_normal_data = convert_to_putracead_format(test_normal_traces)
    test_abnormal_data = convert_to_putracead_format(test_abnormal_traces)
    test_only_latency_abnormal_data = convert_to_putracead_format(test_only_latency_abnormal_traces)
    test_only_structure_abnormal_data = convert_to_putracead_format(test_only_structure_abnormal_traces)
    print('converting to putracead format done!')
    train_datasets = train_normal_data[:int(0.9 * len(train_normal_data))]
    val_datasets = train_normal_data[int(0.9 * len(train_normal_data)):len(train_normal_data)]

    train_total_datasets = copy.deepcopy(train_datasets)
    train_total_datasets.extend(test_abnormal_data[:int(0.6 * len(test_abnormal_data))])
    val_total_datasets = copy.deepcopy(val_datasets)
    val_total_datasets.extend(test_abnormal_data[int(0.6 * len(test_abnormal_data)): int(0.7 * len(test_abnormal_data))])
    test_total_datasets = copy.deepcopy(test_normal_data)
    test_total_datasets.extend(test_abnormal_data[int(0.7 * len(test_abnormal_data)): len(test_abnormal_data)])
    sorted_train_total = sorted(train_total_datasets, key=lambda x: x.get('abnormal'), reverse=True)

    not_test_datasets = test_abnormal_data[:int(0.7 * len(test_abnormal_data))]
    not_test_ids = [not_test_trace['trace_id'] for not_test_trace in not_test_datasets]

    test_latency_datasets = copy.deepcopy(test_normal_data)
    for i in test_only_latency_abnormal_data:
        if i['trace_id'] not in not_test_ids:
            test_latency_datasets.append(i)

    test_structure_datasets = copy.deepcopy(test_normal_data)
    for i in test_only_structure_abnormal_data:
        if i['trace_id'] not in not_test_ids:
            test_structure_datasets.append(i)

    train_ab_num = 0
    for trace in sorted_train_total:
        if trace['abnormal'] == 1:
            train_ab_num += 1

    random.shuffle(val_total_datasets)

    test_ab_num = 0
    for trace in test_total_datasets:
        if trace['abnormal'] == 1:
            test_ab_num += 1
    random.shuffle(test_total_datasets)

    test_latency_ab_num = 0
    for trace in test_latency_datasets:
        if trace['abnormal'] == 1:
            test_latency_ab_num += 1
    random.shuffle(test_latency_datasets)

    test_structure_ab_num = 0
    for trace in test_structure_datasets:
        if trace['abnormal'] == 1:
            test_structure_ab_num += 1
    random.shuffle(test_structure_datasets)

    print(f'total: train_num={len(sorted_train_total)}(abnormal_num:{train_ab_num}), '
          f'val_num={len(val_total_datasets)}, test_total_num={len(test_total_datasets)}(abnormal_num:{test_ab_num}), '
          f'test_latency_num={len(test_latency_datasets)}(abnormal_num:{test_latency_ab_num}, '
          f'test_structure_num={len(test_structure_datasets)}(abnormal_num:{test_structure_ab_num}')

    os.makedirs(os.path.join('data', dataset_name, 'raw'), exist_ok=True)
    os.makedirs(os.path.join('data', dataset_name, 'preprocessed'), exist_ok=True)
    os.makedirs(os.path.join('data', dataset_name, 'processed'), exist_ok=True)

    write_json_to_file(sorted_train_total, 'data/%s/raw/train.json' % dataset_name)
    write_json_to_file(val_total_datasets, 'data/%s/raw/validate.json' % dataset_name)
    write_json_to_file(test_total_datasets, 'data/%s/raw/test_total.json' % dataset_name)
    write_json_to_file(test_latency_datasets, 'data/%s/raw/test_latency.json' % dataset_name)
    write_json_to_file(test_structure_datasets, 'data/%s/raw/test_structure.json' % dataset_name)













