import math
import os
import sys
import traceback
from functools import wraps
from typing import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, recall_score

from .fscore_utils import *

__all__ = ['analyze_anomaly_nll']


def analyze_anomaly_nll(nll_latency: np.ndarray,
                        nll_drop: np.ndarray,
                        label_list: np.ndarray,
                        sample_label_list: List[str] = None,
                        up_sample_normal: int = 1,
                        threshold: Optional[float] = None,
                        method: Optional[str] = None,
                        dataset: Optional[str] = None,
                        save_dict: bool = False,
                        save_filename: str = 'baseline.csv'
                        ) -> Dict[str, float]:
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
    # if up_sample_normal and up_sample_normal > 1:
    #     normal_nll = nll_list[label_list == 0]
    #     normal_label = label_list[label_list == 0]
    #     nll_list = np.concatenate(
    #         [normal_nll] * (up_sample_normal - 1) + [nll_list],
    #         axis=0
    #     )
    #     label_list = np.concatenate(
    #         [normal_label] * (up_sample_normal - 1) + [label_list],
    #         axis=0
    #     )

    # prepare for analyze
    result_dict = {}
    is_anomaly_list = label_list != 0

    # # separated nlls for different labels
    # result_dict['nll_normal'] = float(np.mean(nll_list[label_list == 0]))
    # result_dict['nll_drop'] = float(np.mean(nll_list[label_list == 1]))
    # result_dict['nll_latency'] = float(np.mean(nll_list[label_list == 2]))
    # result_dict['nll_p99'] = float(np.percentile(nll_list, 99))
    #
    # # auc score
    # result_dict['auc'] = float(auc_score(nll_list, is_anomaly_list))

    # nll_drop
    nll_drop_normal = float(np.mean(nll_drop[label_list == 0]))
    nll_drop_anomaly = float(np.mean(nll_drop[label_list == 1]))

    # nll_latency
    nll_latency_normal = float(np.mean(nll_latency[label_list == 0]))
    nll_latency_anomaly = float(np.mean(nll_latency[label_list == 2]))

    # separated nlls for different labels
    # nll_normal is not right, just for test.
    result_dict['nll_normal'] = (nll_drop_normal + nll_latency_normal) / 2
    result_dict['nll_drop'] = nll_drop_anomaly
    result_dict['nll_latency'] = nll_latency_anomaly

    # best f-score
    F = log_error(best_fscore, default_value=(math.nan, math.nan))

    mask_drop = (label_list != 2) & (label_list != 3)
    (best_fscore_drop, thresh_drop, precision_drop, recall_drop,
     TP_drop, TN_drop, FN_drop, FP_drop,
     p_drop, r_drop, f1_drop, acc_drop) = F(nll_drop[mask_drop], label_list[mask_drop] != 0)

    mask_latency = (label_list != 1) & (label_list != 3)
    (best_fscore_latency, thresh_latency, precision_latency, recall_latency,
     TP_latency, TN_latency, FN_latency, FP_latency,
     p_latency, r_latency, f1_latency, acc_latency) = F(nll_latency[mask_latency],
                                                                  label_list[mask_latency] != 0)

    # Total f1_score (anomaly or not, if one is anomaly, then it is anomaly)
    nll_list = nll_latency > thresh_latency
    # nll_list = (nll_drop > thresh_drop) | (nll_latency > thresh_latency)
    (best_fscore_total, threshold, precision, recall, TP, TN, FN, FP,
     p, r, f1, acc) = F(nll_list.astype(np.float32), is_anomaly_list)

    # TP_new, TN_new, FN_new, FP_new = 0, 0, 0, 0
    # for i in range(len(nll_list)):
    #     if nll_list[i] == 1 and is_anomaly_list[i] == 1:
    #         TP_new += 1
    #     elif nll_list[i] == 0 and is_anomaly_list[i] == 0:
    #         TN_new += 1
    #     elif nll_list[i] == 1 and is_anomaly_list[i] == 0:
    #         FP_new += 1
    #     else:
    #         FN_new += 1
    # p_new = round(TP_new / (TP_new + FP_new), 6)
    # r_new = round(TP_new / (TP_new + FN_new), 6)
    # f1_new = round(2 * p_new * r_new / (p_new + r_new), 6)
    # acc_new = round((TP_new + TN_new) / (TP_new + FN_new + FP_new + TN_new), 6)

    result_dict.update({
        # 'best_fscore': round(float(best_fscore_total), 6),
        # 'precision': round(float(precision), 6),
        # 'recall': round(float(recall), 6),
        'TP': TP,
        'TN': TN,
        'FN': FN,
        'FP': FP,
        'p': round(p, 6),
        'r': round(r, 6),
        'f1': round(f1, 6),
        'acc': round(acc, 6),

        # 'TP_new': TP_new,
        # 'TN_new': TN_new,
        # 'FN_new': FN_new,
        # 'FP_new': FP_new,
        # 'p_new': round(p_new, 6),
        # 'r_new': round(r_new, 6),
        # 'f1_new': round(f1_new, 6),
        # 'acc_new': round(acc_new, 6),

        # 'best_fscore_drop': round(float(best_fscore_drop), 6),
        # 'precision_drop': round(float(precision_drop), 6),
        # 'recall_drop': round(float(recall_drop), 6),
        'TP_drop': TP_drop,
        'TN_drop': TN_drop,
        'FN_drop': FN_drop,
        'FP_drop': FP_drop,
        'p_drop': round(p_drop, 6),
        'r_drop': round(r_drop, 6),
        'f1_drop': round(f1_drop, 6),
        'acc_drop': round(acc_drop, 6),

        # 'best_fscore_latency': round(float(best_fscore_latency), 6),
        # 'precision_latency': round(float(precision_latency), 6),
        # 'recall_latency': round(float(recall_latency), 6),
        'TP_latency': TP_latency,
        'TN_latency': TN_latency,
        'FN_latency': FN_latency,
        'FP_latency': FP_latency,
        'p_latency': round(p_latency, 6),
        'r_latency': round(r_latency, 6),
        'f1_latency': round(f1_latency, 6),
        'acc_latency': round(acc_latency, 6),
    })

    # def best_fscore_for_label(label):
    #     not_label = 2 if label == 1 else 1
    #     mask = label_list != not_label
    #     return F(nll_list[mask], label_list[mask] != 0)

    # best_fscore_total, best_threshold, best_pr_total, best_rc_total = F(nll_list, is_anomaly_list)
    # best_fscore_drop, _, best_pr_drop, best_rc_drop = best_fscore_for_label(1)
    # best_fscore_latency, _, best_pr_latency, best_rc_latency = best_fscore_for_label(2)
    # result_dict.update({
    #     'best_fscore': float(best_fscore_total),
    #     'best_fscore_drop': float(best_fscore_drop),
    #     'best_fscore_latency': float(best_fscore_latency),
    #     'best_pr': float(best_pr_total),
    #     'best_rc': float(best_rc_total),
    #     'best_pr_drop': float(best_pr_drop),
    #     'best_rc_drop': float(best_rc_drop),
    #     'best_pr_latency': float(best_pr_latency),
    #     'best_rc_latency': float(best_rc_latency)
    # })
    #
    # # f-score
    # F = log_error(f1_score, default_value=math.nan)
    #
    # def fscore_for_label(label):
    #     not_label = 2 if label == 1 else 1
    #     mask = label_list != not_label
    #     return F(label_list[mask] != 0, nll_list[mask] > threshold)
    #
    # if threshold is not None:
    #     result_dict.update({
    #         'fscore': float(F(is_anomaly_list, nll_list > threshold)),
    #         'fscore_drop': float(fscore_for_label(1)),
    #         'fscore_latency': float(fscore_for_label(2)),
    #     })

    # save result
    if save_dict and method and dataset:
        # dataset = dataset.rstrip('/')

        result_to_save = result_dict.copy()
        result_to_save['dataset'] = dataset
        result_to_save['method'] = method

        if os.path.exists(f'paper-data/{save_filename}'):
            df = pd.read_csv(f'paper-data/{save_filename}')

            if not df[(df['dataset'] == dataset) & (df['method'] == method)].empty:
                df.iloc[df[(df['dataset'] == dataset) & (df['method'] == method)].index[0]] = result_to_save
            else:
                df = df.append(result_to_save, ignore_index=True)
        else:
            df = pd.DataFrame()
            df = df.append(result_to_save, ignore_index=True)

        os.makedirs('paper-data', exist_ok=True)
        df.to_csv(f'paper-data/{save_filename}', index=False)

    if sample_label_list is not None:
        # f1-score for each sample label
        label_dict = []
        label_idx = []

        for l in sample_label_list:
            if l not in label_dict:
                label_dict.append(l)
            label_idx.append(label_dict.index(l))
        
        label_idx = np.array(label_idx)

    return result_dict
