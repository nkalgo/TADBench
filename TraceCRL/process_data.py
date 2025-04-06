import json
import os
import pickle
import random
from queue import Queue
from data_format import *


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


def datetime_to_timestamp(dt):
    # dt = datetime.datetime(2020, 4, 11, 0, 4, 20, 366000)
    timestamp_10 = dt.timestamp()
    timestamp_13 = int(timestamp_10 * 1000) + dt.microsecond // 1000
    return timestamp_13


def tracecrl_preprocess(dataset_name, dataset_dir):
    files = os.listdir(dataset_dir)
    test_res = {}
    for file in files:
        normal_trace_num = 0
        only_latency_abnormal_num = 0
        only_structure_abnormal_num = 0
        both_abnormal_trace_num = 0
        if file not in ['train_normal.pkl', 'test_normal.pkl', 'abnormal.pkl']:
            continue
        print(file)
        data_path = os.path.join(dataset_dir, file)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(len(data))
        random.shuffle(data)
        trace_id_set = set()
        res = {}
        for trace in data:
            trace_id_set.add(trace.trace_id)
            spanId_vertexs_dict = {}
            trace_id = trace.trace_id
            abnormal_flag = [0, 0]
            if trace.anomaly_type == 1:  # latency
                abnormal_flag = [0, 1]
                only_latency_abnormal_num += 1
            elif trace.anomaly_type == 2:  # structure
                abnormal_flag = [1, 0]
                only_structure_abnormal_num += 1
            elif trace.anomaly_type == 3:  # latency & structure
                abnormal_flag = [1, 1]
                both_abnormal_trace_num += 1
            else:
                normal_trace_num += 1

            res[trace_id] = {
                'abnormal': abnormal_flag,
                'vertexs': {'0': 'start'},
                'edges': {}
            }
            spanId_vertexs_dict['0'] = '0'

            spans = get_spans(trace)
            for span in spans:
                if span.span_id in spanId_vertexs_dict.keys():
                    continue
                if span.operation_name is None or span.operation_name == '':
                    span.operation_name = span.service_name
                vertexs_idx = str(len(res[trace_id]['vertexs']))
                res[trace_id]['vertexs'][vertexs_idx] = [span.service_name, span.operation_name, span.anomaly]
                spanId_vertexs_dict[span.span_id] = vertexs_idx

            for span in spans:
                if span.parent_span_id is None or span.parent_span_id == '' or span.parent_span_id == 'nan':
                    span.parent_span_id = '0'
                if spanId_vertexs_dict[span.parent_span_id] not in res[trace_id]['edges'].keys():
                    res[trace_id]['edges'][spanId_vertexs_dict[span.parent_span_id]] = []
                existing_span = [el['spanId'] for el in
                                 res[trace_id]['edges'][spanId_vertexs_dict[span.parent_span_id]]]
                if span.span_id not in existing_span:
                    try:
                        new_edge = {
                            'spanId': span.span_id,
                            'parentSpanId': span.parent_span_id if span.parent_span_id != '0' else '-1',
                            'startTime': datetime_to_timestamp(span.start_time),
                            'service': span.service_name,
                            'operation': span.operation_name,
                            'statusCode': span.status_code,
                            'vertexId': int(spanId_vertexs_dict[span.span_id]),
                            'duration': span.duration
                        }
                    except Exception as e:
                        print(e)
                        print(trace_id)
                        print(span)
                        print(spanId_vertexs_dict)
                        print(res[trace_id]['vertexs'])
                        print(res[trace_id]['edges'])
                        return
                    res[trace_id]['edges'][spanId_vertexs_dict[span.parent_span_id]].append(new_edge)
                else:
                    print('existing span')

        print(f'normal_trace_num={normal_trace_num},\n'
              f'only_latency_abnormal_trace_num={only_latency_abnormal_num},\n'
              f'only_structure_abnormal_trace_num={only_structure_abnormal_num},\n'
              f'both_abnormal_trace_num={both_abnormal_trace_num}')

        print('judge')
        for trace in res.keys():
            id_in_edge = set()
            for id in res[trace]['edges'].keys():
                id_in_edge.add(id)
                for span in res[trace]['edges'][id]:
                    id_in_edge.add(str(span['vertexId']))
            del_vertex = [el for el in list(res[trace]['vertexs'].keys()) if el not in id_in_edge]
            if len(del_vertex):
                print("len(del_vertex)", len(del_vertex), trace)
                for vertex in del_vertex:
                    del res[trace]['vertexs'][vertex]

        items = res.items()
        random.shuffle(list(items))
        res = dict(items)

        if file == 'train_normal.pkl':
            print('len(train_num):', len(res))
            os.mkdir(f'data/{dataset_name}')
            os.mkdir(f'data/{dataset_name}/train')
            os.mkdir(f'data/{dataset_name}/train/preprocessed')
            os.mkdir(f'data/{dataset_name}/train/processed')
            with open(f'data/{dataset_name}/train/preprocessed/train.json', 'w') as res_file:
                json.dump(res, res_file)
        else:
            test_res.update(res)

    items = test_res.items()
    random.shuffle(list(items))
    res = dict(items)
    print('len(test_num):', len(res))
    os.mkdir(f'data/{dataset_name}/test')
    os.mkdir(f'data/{dataset_name}/test/preprocessed')
    os.mkdir(f'data/{dataset_name}/test/processed')
    with open(f'data/{dataset_name}/test/preprocessed/test.json', 'w') as res_file:
        json.dump(res, res_file)

