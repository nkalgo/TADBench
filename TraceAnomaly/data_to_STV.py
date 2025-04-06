import os
import pickle
import shutil
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


def get_trace_from_uniform(filepath):
    with open(filepath, 'rb+') as f:
        print(filepath)
        data = pickle.load(f)
        total_spans = []
        for trace in data:
            total_spans.extend(get_spans(trace))

        count = 0
        traceID = None
        trace = dict()

        for span in total_spans:
            if traceID is None:
                traceID = span.trace_id
            if traceID != span.trace_id:
                count += 1
                if count % 100000 == 0:
                    print('scan count: {}'.format(count))
                yield traceID, trace
                traceID = span.trace_id
                trace = dict()
            # abnormal = 1 if span.anomaly != 0 else 0
            trace[span.span_id] = {'response_time': span.duration,
                                   'operation': f'{span.service_name}/{span.operation_name}',
                                   'start_time': datetime_to_timestamp(span.start_time), 'abnormal': span.anomaly}
            parent = None
            if span.parent_span_id != '' or span.parent_span_id is not None or span.parent_span_id != 'nan':
                parent = span.parent_span_id
            trace[span.span_id]['parent'] = parent
    print("----------span scan done---------")


def walk_call_path(trace, span=None, call_path=''):
    """
    将单个trace转换为call path列表
    """
    # 如果是刚开始 则递归遍历trace中span
    if span == None:
        call_path_list = []
        response_time = []
        for _, span_temp in trace.items():
            call_path_list.append(walk_call_path(trace, span=span_temp))
            response_time.append(span_temp['response_time'])
        return call_path_list, response_time
    # 如果有目标span 则处理该span
    call_path = '#'.join([span['operation'], call_path])
    parent = span['parent']
    # 如果没有父节点 则返回该条callpath 返回时去掉末尾 '#'
    if parent == None:
        return '#'.join(['start', call_path[:-1]])
    elif parent not in trace:
        return '#'.join(['unclear_start', call_path[:-1]])
    # 否则递归处理父节点span
    else:
        return walk_call_path(trace, span=trace[parent], call_path=call_path)


def get_call_path_index(file_list, dataset_name):
    call_path_set = set()
    for file in file_list:
        traces = get_trace_from_uniform(file)
        while True:
            try:
                _, trace = next(traces)
                for cp in walk_call_path(trace)[0]:
                    call_path_set.add(cp)
            except StopIteration:
                break
        # 形成call_path字典
    STV_length = len(call_path_set)
    print('STV_length', STV_length)
    call_path_dict = dict(zip(call_path_set, range(STV_length)))
    pickle.dump(call_path_dict, open('data/%s_idx.pkl' % dataset_name, 'bw'))
    return call_path_dict


def trace_to_STV(trace, call_path_dict, STV_length):
    """
    # 获得trace的STV
    """
    STV = [0] * STV_length
    call_path, response_time = walk_call_path(trace)
    for cp, rt in list(zip(call_path, response_time)):
        if cp not in call_path_dict:
            print('Invalid call path: {}  time:{}'.format(cp, rt))
            return 'Invalid call path: {}  time:{}'.format(cp, rt)
        else:
            if STV[call_path_dict[cp]] < rt:
                STV[call_path_dict[cp]] = rt
    return STV


def adjust_size(filepath, stvsavepath):
    stv = []
    trace_id = []
    with open(filepath, 'r') as wf:
        for info in wf:
            trace_id.append(info.strip().split(':')[0])
            tmp = info.strip().split(':')[1].split(',')
            stv.append(tmp)
        max_len = 0
        for sub in stv:
            if len(sub) > max_len:
                max_len = len(sub)
        for item in stv:
            if len(item) < max_len:
                for i in range(len(item), max_len):
                    item.append(0)
        for sub in stv:
            if len(sub) != max_len:
                print(1)
        for i in range(len(stv)):
            tmp = ",".join(str(item) for item in stv[i])
            stv[i] = tmp

        for i in range(len(trace_id)):
            nf = open(stvsavepath, 'a+')
            nf.write('{}:{}'.format(trace_id[i], stv[i]))
            nf.write('\n')
            nf.close()


def traceanomaly_preprocess(dataset_name, file_dir):
    if os.path.exists('data/%s_STV' % dataset_name):
        shutil.rmtree('data/%s_STV' % dataset_name)
    files = []
    for fi in os.listdir(file_dir):
        if fi not in ['normal.pkl', 'both_abnormal.pkl']:
            files.append(fi)
    os.mkdir('data/%s_STV' % dataset_name)
    # with open('data/%s_idx.pkl' % dataset_name, 'rb') as idx_file:
    #     call_path_dict = pickle.load(idx_file)
    call_path_dict = get_call_path_index([os.path.join(file_dir, file) for file in files], dataset_name)
    STV_length = len(call_path_dict)
    print('STV_length', STV_length)
    for file in files:
        traces = get_trace_from_uniform(os.path.join(file_dir, file))
        print('---%s extract STV---' % file)
        stvsavepath = os.path.join('data/%s_STV' % dataset_name, file.split('.pkl')[0])
        i = 0
        while True:
            try:
                trace_ID, trace = next(traces)
                with open('tmpstvfile', 'a') as f:
                    f.write('{}:{}'.format(trace_ID, ','.join(
                        [str(t) for t in trace_to_STV(trace, call_path_dict, STV_length)])))
                    f.write('\n')
                i += 1
                if i % 10000 == 0:
                    print(i, end=', ')
            except StopIteration:
                break
        print('\n')
        adjust_size('tmpstvfile', stvsavepath)
        if os.path.exists('tmpstvfile'):
            os.remove('tmpstvfile')
