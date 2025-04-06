import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import List

import click


# import sys
# sys.path.append("../../")

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
    source: str = None


def process_data(traces, save_path):
    with open(save_path, 'w') as f:
        f.write("spanId,traceId,parentSpanId,startTime,duration,serviceName,operationName,"
                "status,nodeLatencyLabel,graphLatencyLabel,graphStructureLabel\n")
        for trace in traces:
            q = [trace.root_span]
            while len(q) != 0:
                span = q.pop()
                graphLatencyLabel = None
                graphStructureLabel = None
                if trace.anomaly_type == 0:
                    graphLatencyLabel = 0
                    graphStructureLabel = 0
                elif trace.anomaly_type == 1:
                    graphLatencyLabel = 1
                    graphStructureLabel = 0
                elif trace.anomaly_type == 2:
                    graphLatencyLabel = 0
                    graphStructureLabel = 1
                elif trace.anomaly_type == 3:
                    graphLatencyLabel = 1
                    graphStructureLabel = 1
                f.write(f"{span.span_id},{span.trace_id},{span.parent_span_id},{span.start_time},{span.duration},"
                        f"{span.service_name},{span.operation_name},{span.status_code},{span.anomaly},"
                        f"{graphLatencyLabel}, {graphStructureLabel}\n")
                q.extend(span.children_span_list)


@click.command()
@click.option('-n', '--dataset_name')
@click.option('-dir', '--dataset_dir')
def main(dataset_name, dataset_dir):
    print(dataset_name)
    with open(os.path.join(dataset_dir, 'train_normal.pkl'), 'br') as f:
        train_normal = pickle.load(f)
    train_datasets = train_normal[:int(0.9 * len(train_normal))]
    val_datasets = train_normal[int(0.9 * len(train_normal)):len(train_normal)]

    with open(os.path.join(dataset_dir, 'test_normal.pkl'), 'br') as f:
        test_normal = pickle.load(f)
    with open(os.path.join(dataset_dir, 'abnormal.pkl'), 'br') as f:
        abnormal = pickle.load(f)

    test_datasets = []
    test_datasets.extend(test_normal)
    test_datasets.extend(abnormal)

    print(dataset_name, len(train_datasets), len(val_datasets), len(test_datasets))

    os.mkdir(f'data/{dataset_name}')
    process_data(train_datasets, f'data/{dataset_name}/train.csv')
    process_data(val_datasets, f'data/{dataset_name}/val.csv')
    process_data(test_datasets, f'data/{dataset_name}/test.csv')


if __name__ == '__main__':
    main()
