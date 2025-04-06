import json
from torch_geometric.loader import DataLoader
from model import SIMCLR
from sklearn.svm import OneClassSVM
import torch
from tqdm import tqdm
import numpy as np
import math
import os
import sys
import traceback
from functools import wraps
from typing import *
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, confusion_matrix


def fscore_for_precision_and_recall(precision: np.ndarray,
                                    recall: np.ndarray) -> np.ndarray:
    precision = np.asarray(precision, dtype=np.float64)
    recall = np.asarray(recall, dtype=np.float64)
    return np.where(
        (precision == 0) | (recall == 0),
        0.0,
        2. * np.exp(
            np.log(np.maximum(precision, 1e-8)) +
            np.log(np.maximum(recall, 1e-8)) -
            np.log(np.maximum(precision + recall, 1e-8))
        )
    )


def best_fscore(proba: np.ndarray,
                truth: np.ndarray) -> Tuple[Any, Any, Any, Any]:
    # print(truth)
    # print(proba)
    precision, recall, threshold = precision_recall_curve(truth, proba)
    # print(precision, recall)
    fscore = fscore_for_precision_and_recall(precision, recall)
    idx = np.argmax(fscore[:-1])
    predictions = (proba >= threshold[idx]).astype(int)

    TP = np.sum((predictions == 1) & (truth == 1))
    TN = np.sum((predictions == 0) & (truth == 0))
    FN = np.sum((predictions == 0) & (truth == 1))
    FP = np.sum((predictions == 1) & (truth == 0))

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f1 = 2 * p * r / (p + r)
    acc = (TP + TN) / (TP + FN + FP + TN)

    return fscore[idx], threshold[idx], precision[idx], recall[idx], TP, TN, FN, FP, p, r, f1, acc


def auc_score(proba: np.ndarray, truth: np.ndarray) -> float:
    return float(average_precision_score(truth, proba))


def analyze_anomaly_nll(nll_list: np.ndarray,
                        label_list: np.ndarray,
                        dataset_name: str,
                        up_sample_normal: int = 1):
    def log_error(method, default_value=None):
        @wraps(method)
        def wrapper(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception:
                print(''.join(traceback.format_exception(*sys.exc_info())), file=sys.stderr)
                return default_value

        return wrapper

    def call_plot(fn_, *args, output_file, **kwargs):
        if output_file == ':show:':
            fig = fn_(*args, **kwargs)
            plt.show()
            plt.close()
        else:
            fn_(*args, output_file=output_file, **kwargs)

    # up sample normal nll & label if required
    if up_sample_normal and up_sample_normal > 1:
        normal_nll = nll_list[label_list == 0]
        normal_label = label_list[label_list == 0]
        nll_list = np.concatenate(
            [normal_nll] * (up_sample_normal - 1) + [nll_list],
            axis=0
        )
        label_list = np.concatenate(
            [normal_label] * (up_sample_normal - 1) + [label_list],
            axis=0
        )

    is_anomaly_list = label_list != 0

    # best f-score
    F = log_error(best_fscore, default_value=(math.nan, math.nan))

    (best_fscore_total, threshold, precision, recall, TP, TN, FN, FP,
     p, r, f1, acc) = F(nll_list, is_anomaly_list)

    mask_drop = (label_list != 2) & (label_list != 3)
    (best_fscore_drop, thresh_drop, precision_drop, recall_drop,
     TP_drop, TN_drop, FN_drop, FP_drop,
     p_drop, r_drop, f1_drop, acc_drop) = F(nll_list[mask_drop], label_list[mask_drop] != 0)

    mask_latency = (label_list != 1) & (label_list != 3)
    (best_fscore_latency, thresh_latency, precision_latency, recall_latency,
     TP_latency, TN_latency, FN_latency, FP_latency,
     p_latency, r_latency, f1_latency, acc_latency) = F(nll_list[mask_latency], label_list[mask_latency] != 0)

    for i in [p, r, f1, acc, p_drop, r_drop, f1_drop, acc_drop, p_latency, r_latency, f1_latency, acc_latency]:
        i = round(i, 4)
    print_results = {
        'TP': TP,
        'TN': TN,
        'FN': FN,
        'FP': FP,
        'p': round(p, 6),
        'r': round(r, 6),
        'f1': round(f1, 6),
        'acc': round(acc, 6),

        'TP_drop': TP_drop,
        'TN_drop': TN_drop,
        'FN_drop': FN_drop,
        'FP_drop': FP_drop,
        'p_drop': round(p_drop, 6),
        'r_drop': round(r_drop, 6),
        'f1_drop': round(f1_drop, 6),
        'acc_drop': round(acc_drop, 6),

        'TP_latency': TP_latency,
        'TN_latency': TN_latency,
        'FN_latency': FN_latency,
        'FP_latency': FP_latency,
        'p_latency': round(p_latency, 6),
        'r_latency': round(r_latency, 6),
        'f1_latency': round(f1_latency, 6),
        'acc_latency': round(acc_latency, 6)
    }
    print(print_results)

    result_data = {}
    if os.path.exists('../results.json'):
        with open('../results.json', 'r') as f:
            result_data = json.load(f)

    dataset_results = result_data.setdefault(dataset_name, {})
    algorithm_results = dataset_results.setdefault('TraceCRL', {'total': {}, 'structure': {}, 'latency': {}})
    algorithm_results.update({
        'total': {'p': precision, 'r': recall, 'f1': f1, 'acc': acc},
        'structure': {'p': p_drop, 'r': r_drop, 'f1': f1_drop, 'acc': acc_drop},
        'latency': {'p': p_latency, 'r': r_latency, 'f1': f1_latency, 'acc': acc_latency}
    })

    with open('../results.json', 'w') as f:
        json.dump(result_data, f, indent=4)



def evaluate(test_loader: DataLoader,
             model: SIMCLR,
             dataset_name,
             trace_classifier: OneClassSVM,
             node_classifier: OneClassSVM,
             device=torch.device('cuda:1')):
    trace_nll_list = []
    trace_label_list = []

    # Classify with OC-SVM
    with torch.no_grad():
        for test_traces in tqdm(test_loader):

            test_traces = test_traces.to(device)
            trace_labels = test_traces.y.to(device)

            pred = model(test_traces.x, test_traces.edge_index, test_traces.edge_attr, test_traces.batch)

            # Classify
            cur_trace_nll = trace_classifier.score_samples(pred.detach().cpu().numpy())

            trace_nll_list.extend(cur_trace_nll.tolist())
            for i in range(pred.size(0)):
                if trace_labels[i, 0] == 1 and trace_labels[i, 1] == 0:  # structure
                    trace_label_list.append(1)
                elif trace_labels[i, 0] == 0 and trace_labels[i, 1] == 1:  # latency
                    trace_label_list.append(2)
                elif trace_labels[i, 0] == 1 and trace_labels[i, 1] == 1:  # both
                    trace_label_list.append(3)
                else:
                    trace_label_list.append(0)

    analyze_anomaly_nll(
        nll_list=np.array(trace_nll_list, dtype=np.float32),
        label_list=np.array(trace_label_list, dtype=np.int64),
        dataset_name=dataset_name
    )



