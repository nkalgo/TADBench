import os
import ujson as json
import pickle
import random
from queue import Queue
from data_format import *
from graph import Graph
import logging


def get_spans(trace: Trace):
    spans = []
    root_span = trace.root_span
    q = Queue()
    q.put(root_span)
    while not q.empty():
        span = q.get()
        spans.append(span)
        for child in span.children_span_list:
            q.put(child)
    return spans


def change_uniform_to_crisp(file, res_file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    random.shuffle(data)
    res = []
    '''
    {
        "processes": {
            "S1": {
                "serviceName": "S1"
            },
        },
        "traceID": "A",
        "spans": [
            {
                "traceID": "A",
                "spanID": "B",
                "operationName": "O2",
                "startTime": 10,
                "duration": 70,
                "processID": "S2",
                "references": [
                    {
                        "refType": "CHILD_OF",
                        "traceID": "A",
                        "spanID": "A"
                    }
                ]
            }
        ]
    }
    '''
    i = 0
    for trace in data:
        i += 1
        if i % 10000 == 0:
            print(i, end=',')
        spans = get_spans(trace)
        processes = {}
        new_spans = []
        for span in spans:
            if span.operation_name is None or span.operation_name == '':
                span.operation_name = span.service_name
            processes[span.service_name] = {'serviceName': span.service_name}
            new_span = {
                "traceID": span.trace_id,
                "spanID": span.span_id,
                "operationName": span.operation_name,
                "startTime": int(span.start_time.timestamp() * 1000),
                "duration": span.duration,
                "processID": span.service_name,
                "references": [],
                "abnormal": span.anomaly
            }
            if span != trace.root_span:
                new_span['references'] = [
                    {
                        "refType": "CHILD_OF",
                        "traceID": span.trace_id,
                        "spanID": span.parent_span_id
                    }
                ]
            new_spans.append(new_span)
        res.append({'data': [{'processes': processes, 'traceID': trace.trace_id, 'spans': new_spans}]})
    print('\n')
    with open(res_file, 'w') as f:
        json.dump(res, f)


def get_call_path_index(dir_name, res_pkl_name):
    files = os.listdir(dir_name)
    call_path_set = set()
    debug_on = logging.getLogger(__name__).isEnabledFor(logging.DEBUG)
    for file in files:
        print(file)
        with open(os.path.join(dir_name, file), 'r') as f:
            data = json.load(f)
        print(len(data))
        for trace in data:
            spans = trace['data'][0]['spans']
            for span in spans:
                if not span['references']:
                    serviceName = span['processID']
                    operationName = span['operationName']
                    break
            graph = Graph(trace, serviceName, operationName, file, True)
            res = graph.findCriticalPath()
            debug_on and logging.debug("critical path:" + str(res))
            metrics = graph.getMetrics(res)
            debug_on and logging.debug(metrics.opTimeExclusive)
            debug_on and logging.debug("Test result = " + str(graph.checkResults(metrics.opTimeExclusive)))
            paths = list(metrics.callpathTimeExclusive.keys())
            for path in paths:
                call_path_set.add(path)
        call_path_dict = dict(zip(call_path_set, range(len(call_path_set))))
    print(len(call_path_dict))
    with open(res_pkl_name, 'bw') as f:
        pickle.dump(call_path_dict, f)


def generate_SCPV(file, savepath, labelsavepath, pkl_name):
    with open(pkl_name, 'br') as f:
        call_path_dict = pickle.load(f)
        SCPV_length = len(call_path_dict)
    debug_on = logging.getLogger(__name__).isEnabledFor(logging.DEBUG)
    with open(file, 'r') as f:
        data = json.load(f)
    for trace in data:
        SCPV = [0] * SCPV_length
        SCPV_label = [0] * SCPV_length
        spans = trace['data'][0]['spans']
        for span in spans:
            if not span['references']:
                serviceName = span['processID']
                operationName = span['operationName']
                break
        graph = Graph(trace, serviceName, operationName, file, True)
        res = graph.findCriticalPath()
        debug_on and logging.debug("critical path:" + str(res))

        metrics = graph.getMetrics(res)
        debug_on and logging.debug(metrics.opTimeExclusive)
        debug_on and logging.debug("Test result = " + str(graph.checkResults(metrics.opTimeExclusive)))
        path_abnormal_dict = graph.getCriticalPathWithAbnormal(res)
        path_excl_dict = metrics.callpathTimeExclusive
        for path in path_excl_dict.keys():
            if path not in call_path_dict:
                logging.error('Invalid call path: {}'.format(path))
                return
            else:
                if SCPV[call_path_dict[path]] < path_excl_dict[path]:
                    SCPV[call_path_dict[path]] = path_excl_dict[path]

        for path in path_abnormal_dict.keys():
            if path not in call_path_dict:
                logging.error('Invalid call path: {}'.format(path))
                return
            else:
                if SCPV_label[call_path_dict[path]] < path_abnormal_dict[path]:
                    SCPV_label[call_path_dict[path]] = path_abnormal_dict[path]

        with open(savepath, 'a') as f:
            f.write('{}:{}'.format(trace['data'][0]['traceID'], ','.join(
                [str(t) for t in SCPV])))
            f.write('\n')

        with open(labelsavepath, 'a') as f:
            f.write('{}:{}'.format(trace['data'][0]['traceID'], ','.join(
                [str(t) for t in SCPV_label])))
            f.write('\n')


def crisp_preprocess(dataset_name, data_path):
    save_dir = f'data/{dataset_name}'
    os.mkdir(save_dir)
    files = []
    for file in os.listdir(data_path):
        if file not in ['both_abnormal.pkl', 'normal.pkl']:
            files.append(file)
    for file in files:
        print(file)
        change_uniform_to_crisp(os.path.join(data_path, file), '%s/%s.json' % (save_dir, file.split('.pkl')[0]))

    get_call_path_index(f'data/{dataset_name}', f'data/{dataset_name}.pkl')

    dataset_dir = f'data/{dataset_name}'
    save_dir = f'data/{dataset_name}_SCPV'
    os.mkdir(save_dir)
    files = os.listdir(dataset_dir)
    for file in files:
        save_dataset_name = file.split('.json')[0]
        generate_SCPV(os.path.join(dataset_dir, file), os.path.join(save_dir, save_dataset_name),
                      os.path.join(save_dir, '%s_label' % save_dataset_name), f'data/{dataset_name}.pkl')
