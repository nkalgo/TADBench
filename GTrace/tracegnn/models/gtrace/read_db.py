import random
from queue import Queue
from typing import *

import yaml
from tqdm import tqdm

from .config import ExpConfig
import mltk
import dgl
from loguru import logger
import torch
import torch.backends.cudnn
import os
import numpy as np

from tracegnn.data import *
from tracegnn.utils import *

from .dataset import TestDataset, TrainDataset

from .trainer import trainer

import os
import pandas as pd
import csv
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import *
from typing import Tuple, List


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str  # 如果没有，则为None
    children_span_list: List['Span']

    start_time: datetime
    duration: float  # 单位为毫秒
    service_name: str
    anomaly: int = 0  # normal:0/anomaly:1
    status_code: str = None

    operation_name: str = None
    root_cause: bool = None  # True
    latency: int = None  # normal:None/anomaly:1
    structure: int = None  # normal:None/anomaly:1
    extra: dict = None


@dataclass
class Trace:
    trace_id: str
    root_span: Span
    span_count: int = 0
    anomaly_type: int = None  # normal:0/only_latency_anomaly:1/only_structure_anomaly:2/both_anomaly:3
    source: str = None  # AIOps2020 / AIOps2021 / Hipster-shop / TrainTicket / AIOps2022 / AIOps2023 / GAIA

    def __repr__(self):
        return (f"Trace(trace_id='{self.trace_id}', anomaly_type={self.anomaly_type}, source='{self.source}', "
                f"root_span={self.root_span}, span_count={self.span_count})")


def print_trace(trace):
    print(
        f"Trace(trace_id='{trace.trace_id}', anomaly_type={trace.anomaly_type}, root_span={print_span(trace.root_span, level=0)})")


def print_span(span, level=0):
    indent = "  " * level
    output = f"{indent}Span(span_id='{span.span_id}', parent_span_id={span.parent_span_id}, children_span_list=[\n"
    for child in span.children_span_list:
        output += print_span(child, level + 1) + ",\n"
    output += f"{indent}]"
    output += f"op_name={span.operation_name}, anomaly={span.anomaly}"
    return output


def get_datasets(dataset):
    trace_id_to_spans = {}
    traces = []
    struct_trace_id_set = set()
    for i in tqdm(dataset):
        trace_id, spans, struct_trace_id_set = convert_trace_graph_to_spans(i, struct_trace_id_set)
        # 以下是去重span
        if trace_id not in trace_id_to_spans.keys():
            trace_id_to_spans[trace_id] = []
        exist_span_id = set()
        for exist_span in trace_id_to_spans[trace_id]:
            exist_span_id.add(exist_span.span_id)
        for span in spans:
            if span.span_id not in exist_span_id:
                exist_span_id.add(span.span_id)
                trace_id_to_spans[trace_id].append(span)
    print('struct_trace_id_set', struct_trace_id_set)
    print('len(struct_trace_id_set)', len(struct_trace_id_set))

    for trace_id in trace_id_to_spans.keys():
        spans = trace_id_to_spans[trace_id]
        span_dict = {span.span_id: span for span in spans}
        root_span = None
        for span in spans:
            if span.parent_span_id is None:
                if root_span:
                    # print(trace_id, '有多个根节点')
                    # for s in spans:
                    #     print(s.parent_span_id, s.span_id)
                    # break
                    span_count = 0
                    q = Queue()
                    q.put(span)
                    while not q.empty():
                        k = q.get()
                        span_count += 1
                        for child in k.children_span_list:
                            q.put(child)

                    span_count_0 = 0
                    q = Queue()
                    q.put(root_span)
                    while not q.empty():
                        k = q.get()
                        span_count_0 += 1
                        for child in k.children_span_list:
                            q.put(child)

                    if span_count > span_count_0:
                        root_span = span
                else:
                #     trace = Trace(trace_id=trace_id, root_span=span, source='Google')
                #     traces.append(trace)
                    root_span = span
            else:
                parent_span = span_dict.get(span.parent_span_id, None)
                if parent_span:
                    parent_span.children_span_list.append(span)
                    # print(len(parent_span.children_span_list))
                else:
                    # 找不到父节点，就把父节点id设为None，并判断它是否可以成为根节点
                    span.parent_span_id = None
                    if root_span:
                        # print(trace_id, '有多个根节点')
                        span_count = 0
                        q = Queue()
                        q.put(span)
                        while not q.empty():
                            k = q.get()
                            span_count += 1
                            for child in k.children_span_list:
                                q.put(child)

                        span_count_0 = 0
                        q = Queue()
                        q.put(root_span)
                        while not q.empty():
                            k = q.get()
                            span_count_0 += 1
                            for child in k.children_span_list:
                                q.put(child)

                        if span_count > span_count_0:
                            root_span = span
                            print(root_span)
                            print('\n\n\n')
                    else:
                        #     trace = Trace(trace_id=trace_id, root_span=span, source='Google')
                        #     traces.append(trace)
                        root_span = span
                    # if span.structure == 1:
                    # print(span.parent_span_id, span.span_id)
                    # for d in spans:
                    #     if d.structure == 1:
                    #         print('找不到父节点')
                    #         for s in spans:
                    #             print(s.parent_span_id, s.span_id, s.structure)
                    #         break
        if trace_id == '9923031945910562291.10559265709303171512':
            print('root_span', root_span.span_id)
            for span in spans:
                print(span.parent_span_id, span.span_id, span.structure)

        if root_span:
            span_count = 0
            latency_abnormal = False
            structure_abnormal = False
            q = Queue()
            q.put(root_span)
            while not q.empty():
                span = q.get()
                # print(span.parent_span_id, span.span_id, span.structure)
                '''span_count: int = 0
    anomaly_type: int = None  # normal:0/only_latency_anomaly:1/only_structure_anomaly:2/both_anomaly:3'''
                span_count += 1
                if span.latency == 1:
                    latency_abnormal = True
                if span.structure == 1:
                    structure_abnormal = True

                for child in span.children_span_list:
                    q.put(child)

            trace_abnormal = 0
            if not latency_abnormal and not structure_abnormal:
                trace_abnormal = 0
            elif latency_abnormal and not structure_abnormal:
                trace_abnormal = 1
            elif not latency_abnormal and structure_abnormal:
                trace_abnormal = 2
            elif latency_abnormal and structure_abnormal:
                trace_abnormal = 3
            trace = Trace(trace_id=trace_id, root_span=root_span, span_count=span_count, anomaly_type=trace_abnormal, source='Google')
            if trace_id == '9923031945910562291.10559265709303171512':
                print(span_count, trace_abnormal)
            traces.append(trace)

    # for trace in traces:
    #     if trace_id == '459535774061927953.4867060530947287969':
    #         print(trace)



    normal_traces = []
    only_latency_abnormal_traces = []
    only_structure_abnormal_traces = []
    both_abnormal_traces = []
    abnormal_traces = []
    trace_dict = {}
    for trace in tqdm(traces):
        # trace = convert_trace_graph_to_span(i)
        # if trace.trace_id in trace_dict.keys():
        #     print(trace_dict[trace.trace_id])
        #     print('\n\n\n')
        #     print(trace)
        #     print('\n\n\n')
        #     exit()
        # else:
        #     trace_dict[trace.trace_id] = trace
        if trace.anomaly_type == 0:
            normal_traces.append(trace)
        elif trace.anomaly_type == 1:
            only_latency_abnormal_traces.append(trace)
            abnormal_traces.append(trace)
        elif trace.anomaly_type == 2:
            only_structure_abnormal_traces.append(trace)
            abnormal_traces.append(trace)
        else:
            both_abnormal_traces.append(trace)
            abnormal_traces.append(trace)

    print(f'normal_trace_num={len(normal_traces)},\n'
          f'only_latency_abnormal_trace_num={len(only_latency_abnormal_traces)},\n'
          f'only_structure_abnormal_trace_num={len(only_structure_abnormal_traces)},\n'
          f'both_abnormal_trace_num={len(both_abnormal_traces)},\nabnormal_trace_num={len(abnormal_traces)}\n')

    return (normal_traces, only_latency_abnormal_traces, only_structure_abnormal_traces,
            both_abnormal_traces, abnormal_traces)


def main(exp: mltk.Experiment):
    # config
    config: ExpConfig = exp.config
    # train_dataset = TrainDataset(config, valid=False).get_traces()  # 49996
    # (normal_traces, only_latency_abnormal_traces, only_structure_abnormal_traces,
    #  both_abnormal_traces, abnormal_traces) = get_datasets(train_dataset)

    # dst_dir = '../Datasets/Google'
    #
    # random.shuffle(normal_traces)
    # with open(os.path.join(dst_dir, 'train_normal.pkl'), 'wb') as f:
    #     pickle.dump(normal_traces, f)

    # val_normal_traces, val_only_structure_abnormal_traces, val_only_structure_abnormal_traces, val_both_abnormal_traces, val_abnormal_traces = get_datasets(
    #     val_dataset)
    # normal_traces.extend(val_normal_traces)
    test_dataset = TestDataset(config).get_traces()  # 20000
    for trace in test_dataset:
        if trace.parent_id == 0 or not isinstance(trace.parent_id, int):
            print(trace)
            break
    # trace_id_to_spans = {}
    # traces = []
    # exist_trace_id = set()
    # # for i in tqdm(test_dataset):
    #     # trace = convert_root_trace_graph_to_uniform(i)
    # # val_dataset = TrainDataset(config, valid=True).get_traces()  # 4997
    # # test_dataset.extend(val_dataset)
    # # test_dataset.extend(train_dataset)
    # (normal_traces, only_latency_abnormal_traces, only_structure_abnormal_traces,
    #  both_abnormal_traces, abnormal_traces) = get_datasets(test_dataset)
    #
    # # random.shuffle(only_structure_abnormal_traces)
    # # with open(os.path.join(dst_dir, 'only_structure_abnormal.pkl'), 'wb') as f:
    # #     pickle.dump(only_structure_abnormal_traces, f)
    # #
    # # random.shuffle(only_latency_abnormal_traces)
    # # with open(os.path.join(dst_dir, 'only_latency_abnormal.pkl'), 'wb') as f:
    # #     pickle.dump(only_latency_abnormal_traces, f)
    # #
    # # random.shuffle(both_abnormal_traces)
    # # with open(os.path.join(dst_dir, 'both_abnormal.pkl'), 'wb') as f:
    # #     pickle.dump(both_abnormal_traces, f)
    # #
    # # random.shuffle(abnormal_traces)
    # # with open(os.path.join(dst_dir, 'abnormal.pkl'), 'wb') as f:
    # #     pickle.dump(abnormal_traces, f)
    # #
    # # random.shuffle(normal_traces)
    # # with open(os.path.join(dst_dir, 'test_normal.pkl'), 'wb') as f:
    # #     pickle.dump(normal_traces, f)





'''
class TraceGraph(object):
    trace_id: Optional[Tuple[int, int]] # (1289, 1562)
    parent_id: Optional[int]  # 569745
    root: TraceGraphNode
    node_count: Optional[int] # 26
    max_depth: Optional[int]  # 4
    anomaly: int  # 0 normal, 1 drop, 2 latency  # None
    status: Set[str] '0'
    
    
class TraceGraphNode(object):
    node_id: Optional[int]  # the node id of the graph 0
    service_id: Optional[int]  # the service id             # 6
    status_id: Optional[int]                                # 1
    operation_id: int  # the operation id                   # 30
    features: TraceGraphNodeFeatures  # the node features
    children: List['TraceGraphNode']  # children nodes      # []
    spans: Optional[List[TraceGraphSpan]]  # detailed spans information (from the original data)
    scores: Optional[TraceGraphNodeReconsScores]
    anomaly: Optional[int]  # 1: drop anomaly; 2: latency anomaly; 3: service type anomaly   # None

class TraceGraphSpan(object):
    span_id: Optional[int]
    start_time: Optional[datetime]  # datetime.datetime(2022, 4, 13, 18, 18, 13, 883245)
    latency: float 2.915
    status: str '0'
'''


def convert_root_trace_graph_to_uniform(trace_graph: 'TraceGraph') -> Trace:
    trace_id = '%s.%s' % (trace_graph.trace_id[0], trace_graph.trace_id[1])
    with open('dataset/' + 'dataset_b' + '/processed/operation_id.yml', 'r') as f:
        operation_id_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open('dataset/' + 'dataset_b' + '/processed/service_id.yml', 'r') as f:
        service_id_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open('dataset/' + 'dataset_b' + '/processed/status_id.yml', 'r') as f:
        status_id_dict = yaml.load(f, Loader=yaml.FullLoader)
    operation_id_dict = {int(v): k for k, v in operation_id_dict.items()}
    service_id_dict = {int(v): k for k, v in service_id_dict.items()}
    latency_abnormal = False
    structure_abnormal = False
    span_num = 0

    # 递归函数来构建Span树
    def build_span_tree(node: 'TraceGraphNode', parent_span_id: Optional[str] = None) -> Span:
        nonlocal structure_abnormal, latency_abnormal, span_num
        span_id = str(node.spans[0].span_id)
        anomaly = 0
        latency = None
        structure = None
        if node.anomaly is None or node.anomaly == 0:
            anomaly = 0
        elif node.anomaly == 1:
            anomaly = 1
            structure = 1
            structure_abnormal = True
        elif node.anomaly == 2:
            anomaly = 1
            latency = 1
            latency_abnormal = True
        if len(node.spans) > 1:
            print('node.spans', len(node.spans))
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            children_span_list=list(),
            start_time=node.spans[0].start_time,
            duration=node.spans[0].latency * 1000,
            service_name=service_id_dict[node.service_id],
            anomaly=anomaly,
            status_code='%s00' % status_id_dict[node.spans[0].status],
            operation_name=operation_id_dict[node.operation_id],
            root_cause=None,
            latency=latency,
            structure=structure
        )
        span_num += 1
        for child_node in node.children:
            child_span = build_span_tree(child_node, span_id)
            span.children_span_list.append(child_span)

        return span

    # 从根节点开始构建Span树
    root_span = build_span_tree(trace_graph.root)

    trace_abnormal = 0
    if not latency_abnormal and not structure_abnormal:
        trace_abnormal = 0
    elif latency_abnormal and not structure_abnormal:
        trace_abnormal = 1
    elif not latency_abnormal and structure_abnormal:
        trace_abnormal = 2
        # print(trace_id, root_span)
    elif latency_abnormal and structure_abnormal:
        trace_abnormal = 3
    # 创建Trace对象
    trace = Trace(trace_id=trace_id, root_span=root_span, span_count=span_num, anomaly_type=trace_abnormal,
                  source='Google')
    return trace


def convert_trace_graph_to_spans(trace_graph: 'TraceGraph', struct_trace_id_set):
    trace_id = '%s.%s' % (trace_graph.trace_id[0], trace_graph.trace_id[1])
    with open('dataset/' + 'dataset_b' + '/processed/operation_id.yml', 'r') as f:
        operation_id_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open('dataset/' + 'dataset_b' + '/processed/service_id.yml', 'r') as f:
        service_id_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open('dataset/' + 'dataset_b' + '/processed/status_id.yml', 'r') as f:
        status_id_dict = yaml.load(f, Loader=yaml.FullLoader)
    operation_id_dict = {int(v): k for k, v in operation_id_dict.items()}
    service_id_dict = {int(v): k for k, v in service_id_dict.items()}

    spans = []

    def get_span(node: 'TraceGraphNode', parent_span_id: Optional[str] = None):
        nonlocal spans
        span_id = str(node.spans[0].span_id)
        anomaly = 0
        latency = None
        structure = None
        if node.anomaly is None or node.anomaly == 0:
            anomaly = 0
        elif node.anomaly == 1:
            anomaly = 1
            structure = 1
        elif node.anomaly == 2:
            anomaly = 1
            latency = 1
        if len(node.spans) > 1:
            print('node.spans', len(node.spans))
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            children_span_list=list(),
            start_time=node.spans[0].start_time,
            duration=node.spans[0].latency * 1000,
            service_name=service_id_dict[node.service_id],
            anomaly=anomaly,
            status_code='%s00' % status_id_dict[node.spans[0].status],
            operation_name=operation_id_dict[node.operation_id],
            root_cause=None,
            latency=latency,
            structure=structure
        )
        if structure == 1:
            struct_trace_id_set.add(trace_id)
        spans.append(span)
        for child_node in node.children:
            get_span(child_node, span_id)

    get_span(trace_graph.root, str(trace_graph.parent_id) if str(trace_graph.parent_id) != '0' else None)

    return trace_id, spans, struct_trace_id_set


if __name__ == '__main__':
    with mltk.Experiment(ExpConfig) as exp:
        main(exp)

    # traces = []
    # exist_trace_id = set()
    # trace_id = '0'
    # spans = [Span(trace_id='0', span_id='b', parent_span_id='a', start_time=None, duration=0, service_name='aa',
    #               children_span_list=[
    #                   Span(trace_id='0', span_id='c', parent_span_id='b', start_time=None, duration=0,
    #                        service_name='aa',
    #                        children_span_list=[])]
    #               ),
    #          Span(trace_id='0', span_id='e', parent_span_id=None, start_time=None, duration=0, service_name='aa',
    #               children_span_list=[]
    #               ),
    #          Span(trace_id='0', span_id='f', parent_span_id='e', start_time=None, duration=0, service_name='aa',
    #               children_span_list=[
    #                   Span(trace_id='0', span_id='g', parent_span_id='f', start_time=None, duration=0,
    #                        service_name='aa',
    #                        children_span_list=[])]
    #               ),
    #          Span(trace_id='0', span_id='a', parent_span_id='g', start_time=None, duration=0, service_name='aa',
    #               children_span_list=[]
    #               )
    #          ]
    # span_dict = {span.span_id: span for span in spans}
    # # 构建 Span 树
    # for span in spans:
    #     if span.parent_span_id:
    #         parent_span = span_dict.get(span.parent_span_id)
    #         if parent_span:
    #             parent_span.children_span_list.append(span)
    #
    # # 查找根节点并创建 Trace 对象
    # for span in spans:
    #     if span.parent_span_id is None:
    #         if trace_id in exist_trace_id:
    #             print(trace_id, '有多个根节点')
    #             break
    #         else:
    #             trace = Trace(trace_id=trace_id, root_span=span, source='Google')
    #             traces.append(trace)
    #             exist_trace_id.add(trace_id)
    # # for span in spans:
    # #     if span.parent_span_id is None:
    # #         if trace_id in exist_trace_id:
    # #             print(trace_id, '有多个根节点')
    # #             break
    # #         else:
    # #             trace = Trace(trace_id=trace_id, root_span=span, source='Google')
    # #             traces.append(trace)
    # #             exist_trace_id.add(trace_id)
    # #         continue
    # #     for other_span in spans:
    # #         if span.parent_span_id == other_span.span_id:
    # #             other_span.children_span_list.append(span)
    # #             break
    # print(traces[0])
