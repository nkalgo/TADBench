from typing import *

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from tracegnn.utils.fscore_utils import *
import os

# __all__ = ['analyze_anomaly_nll']

from tracegnn.utils.fscore_utils import best_fscore_with_counts, new_fscore


def cal(TP, TN, FN, FP):
    # TP, FP, TN, FN = 0, 0, 0, 0
    # scores.sort()
    # threshold = scores[len(scores) // 100 * 15]  # 选择排在前 15% 的得分作为阈值
    # # print(scores)
    # # print(threshold)
    # predict = []
    # for score in scores:
    #     if score > threshold:
    #         predict.append(0)
    #     else:
    #         predict.append(1)
    #     # predict.append(0 if score > threshold else 1)
    # # print(predict)
    # # print(labels)
    # labels = labels.astype(int)
    # # for p in labels:
    # #     if p == 1:
    # #         print('yes', end=', ')
    # for label, p in zip(labels, predict):
    #     # if label in [0, 1]:
    #     #     print(label, end=', ')
    #     if label == 1 and p == 1:
    #         # print('tp', end=', ')
    #         TP += 1
    #     elif label == 0 and p == 0:
    #         TN += 1
    #     elif label == 0 and p == 1:
    #         # print('fp', end=', ')
    #         FP += 1
    #     elif label == 1 and p == 0:
    #         FN += 1
    # print(TP, FP, TN, FN)
    # # F1_score = f1_score(original, predict, average='binary')
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)
    ACC = (TP + TN) / (TP + FN + FP + TN)
    print("TP: {}".format(TP), end=', ')
    print("TN: {}".format(TN), end=', ')
    print("FP: {}".format(FP), end=', ')
    print("FN: {}".format(FN))
    print("precision: {}".format(precision), end=', ')
    print("recall: {}".format(recall), end=', ')
    print("F1_score: {}".format(F1_score), end=', ')
    print('ACC:{}'.format(ACC))


def analyze_multiclass_anomaly_nll(nll_latency: np.ndarray,
                        nll_drop: np.ndarray,
                        label_list: np.ndarray,
                        node_latency: np.array,
                        node_label_list: np.array,
                        method: Optional[str] = None,
                        dataset: Optional[str] = None,
                        save_dict: bool = False,
                        save_filename: str='baseline.csv'
                        ) -> Dict[str, float]:
    # prepare for analyze
    result_dict = {}
    is_anomaly_list = label_list != 0
    drop_anomaly_list = label_list == 1
    latency_anomaly_list = label_list == 2

    is_node_anomaly_list = node_label_list != 0

    # nll_drop
    nll_drop_normal = float(np.mean(nll_drop[label_list == 0]))
    nll_drop_anomaly = float(np.mean(nll_drop[label_list == 1]))

    # nll_latency
    nll_latency_normal = float(np.mean(nll_latency[label_list == 0]))
    nll_latency_anomaly = float(np.mean(nll_latency[label_list == 2]))

    node_latency_normal = float(np.mean(node_latency[node_label_list == 0]))
    node_latency_anomaly = float(np.mean(node_latency[node_label_list == 1]))

    # separated nlls for different labels
    # nll_normal is not right, just for test.
    result_dict['nll_normal'] = (nll_drop_normal + nll_latency_normal) / 2
    result_dict['nll_drop'] = nll_drop_anomaly
    result_dict['nll_latency'] = nll_latency_anomaly
    result_dict['node_latency'] = node_latency_anomaly


    # drop and latency auc score
    drop_auc = float(auc_score(nll_drop, drop_anomaly_list))
    latency_auc = float(auc_score(nll_latency, latency_anomaly_list))
    # This auc value is not right, just for test.
    result_dict['auc'] = (drop_auc + latency_auc) / 2

    # Find threshold
    # best_fscore_drop, thresh_drop, precision_drop, recall_drop, TP_drop, TN_drop, FN_drop, FP_drop = best_fscore(nll_drop, drop_anomaly_list)
    # print('drop')
    # cal(TP_drop, TN_drop, FN_drop, FP_drop)
    # best_fscore_latency, thresh_latency, precision_latency, recall_latency, TP_latency, TN_latency, FN_latency, FP_latency = best_fscore(nll_latency, latency_anomaly_list)
    # print('latency')
    # cal(TP_latency, TN_latency, FN_latency, FP_latency)

    # (best_fscore_drop, thresh_drop, precision_drop, recall_drop,
    #  TP_drop, TN_drop, FN_drop, FP_drop,
    #  p_drop, r_drop, f1_drop, acc_drop) = best_fscore(
    #     nll_drop, drop_anomaly_list)
    #
    # (best_fscore_latency, thresh_latency, precision_latency, recall_latency,
    #  TP_latency, TN_latency, FN_latency, FP_latency,
    #  p_latency, r_latency, f1_latency, acc_latency) = best_fscore(
    #     nll_latency, latency_anomaly_list)

    mask_drop = (label_list != 2) & (label_list != 3)
    (best_fscore_drop, thresh_drop, precision_drop, recall_drop,
     TP_drop, TN_drop, FN_drop, FP_drop,
     p_drop, r_drop, f1_drop, acc_drop) = best_fscore(nll_drop[mask_drop], label_list[mask_drop] != 0)

    # (best_fscore_drop_new, thresh_drop_new, precision_drop_new, recall_drop_new,
    #  TP_drop_new, TN_drop_new, FN_drop_new, FP_drop_new,
    #  p_drop_new, r_drop_new, f1_drop_new, acc_drop_new) = new_fscore(nll_drop[mask_drop], label_list[mask_drop] != 0)

    mask_latency = (label_list != 1) & (label_list != 3)
    (best_fscore_latency, thresh_latency, precision_latency, recall_latency,
     TP_latency, TN_latency, FN_latency, FP_latency,
     p_latency, r_latency, f1_latency, acc_latency) = best_fscore(nll_latency[mask_latency], label_list[mask_latency] != 0)

    # (best_fscore_latency_new, thresh_latency_new, precision_latency_new, recall_latency_new,
    #  TP_latency_new, TN_latency_new, FN_latency_new, FP_latency_new,
    #  p_latency_new, r_latency_new, f1_latency_new, acc_latency_new) = new_fscore(nll_latency[mask_latency],
    #                                                               label_list[mask_latency] != 0)

    (best_fscore_node_latency, thresh_node_latency, precision_node_latency, recall_node_latency,
     TP_node_latency, TN_node_latency, FN_node_latency, FP_node_latency,
     p_node_latency, r_node_latency, f1_node_latency, acc_node_latency) = best_fscore(node_latency, is_node_anomaly_list)

    # (best_fscore_node_latency_new, thresh_node_latency_new, precision_node_latency_new, recall_node_latency_new,
    #  TP_node_latency_new, TN_node_latency_new, FN_node_latency_new, FP_node_latency_new,
    #  p_node_latency_new, r_node_latency_new, f1_node_latency_new, acc_node_latency_new) = new_fscore(node_latency,
    #                                                                                   is_node_anomaly_list)

    # Total f1_score (anomaly or not, if one is anomaly, then it is anomaly)
    nll_list = nll_latency > thresh_latency
    # best_fscore_total, threshold, precision, recall, TP, TN, FN, FP = best_fscore(nll_list.astype(np.float32), is_anomaly_list)
    # print('total')
    # cal(TP, TN, FN, FP)

    (best_fscore_total, threshold, precision, recall, TP, TN, FN, FP,
     p, r, f1, acc) = best_fscore(nll_list.astype(np.float32), is_anomaly_list)

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
    # (best_fscore_total_new, threshold_new, precision_new, recall_new, TP_new, TN_new, FN_new, FP_new,
    #  p_new, r_new, f1_new, acc_new) = new_fscore(nll_list.astype(np.float32), is_anomaly_list)

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

        # 'TP_drop_new': TP_drop_new,
        # 'TN_drop_new': TN_drop_new,
        # 'FN_drop_new': FN_drop_new,
        # 'FP_drop_new': FP_drop_new,
        # 'p_drop_new': round(p_drop_new, 6),
        # 'r_drop_new': round(r_drop_new, 6),
        # 'f1_drop_new': round(f1_drop_new, 6),
        # 'acc_drop_new': round(acc_drop_new, 6),

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

        # 'TP_latency_new': TP_latency_new,
        # 'TN_latency_new': TN_latency_new,
        # 'FN_latency_new': FN_latency_new,
        # 'FP_latency_new': FP_latency_new,
        # 'p_latency_new': round(p_latency_new, 6),
        # 'r_latency_new': round(r_latency_new, 6),
        # 'f1_latency_new': round(f1_latency_new, 6),
        # 'acc_latency_new': round(acc_latency_new, 6),

        'TP_node_latency': TP_node_latency,
        'TN_node_latency': TN_node_latency,
        'FN_node_latency': FN_node_latency,
        'FP_node_latency': FP_node_latency,
        'p_node_latency': round(p_node_latency, 6),
        'r_node_latency': round(r_node_latency, 6),
        'f1_node_latency': round(f1_node_latency, 6),
        'acc_node_latency': round(acc_node_latency, 6),

        # 'TP_node_latency_new': TP_node_latency_new,
        # 'TN_node_latency_new': TN_node_latency_new,
        # 'FN_node_latency_new': FN_node_latency_new,
        # 'FP_node_latency_new': FP_node_latency_new,
        # 'p_node_latency_new': round(p_node_latency_new, 6),
        # 'r_node_latency_new': round(r_node_latency_new, 6),
        # 'f1_node_latency_new': round(f1_node_latency_new, 6),
        # 'acc_node_latency_new': round(acc_node_latency_new, 6),
    })

    # print('another total')
    # print('previous threshold', threshold)
    # fscore, threshold, TP, TN, FP, FN = best_fscore_with_counts(nll_list.astype(np.float32), is_anomaly_list)
    #
    # print('fscore, threshold, TP, TN, FP, FN', fscore, threshold, TP, TN, FP, FN)
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # F1_score = 2 * precision * recall / (precision + recall)
    # ACC = (TP + TN) / (TP + FN + FP + TN)
    # print("precision: {}".format(precision), end=', ')
    # print("recall: {}".format(recall), end=', ')
    # print("F1_score: {}".format(F1_score), end=', ')
    # print('ACC:{}'.format(ACC))

    # result_dict.update({
    #     'best_fscore': float(best_fscore_total),
    #     'best_fscore_drop': float(best_fscore_drop),
    #     'best_fscore_latency': float(best_fscore_latency),
    #     'precision':float(precision),
    #     'recall':float(recall)
    # })

    if save_dict and method and dataset:
        # dataset = dataset.rstrip('/')

        result_to_save = result_dict.copy()
        result_to_save['dataset'] = dataset
        result_to_save['method'] = method

        if os.path.exists(f'lstm_res/{save_filename}'):
            df = pd.read_csv(f'lstm_res/{save_filename}')

            if not df[(df['dataset'] == dataset) & (df['method'] == method)].empty:
                df.iloc[df[(df['dataset'] == dataset) & (df['method'] == method)].index[0]] = result_to_save
            else:
                df = df.append(result_to_save, ignore_index=True)
        else:
            df = pd.DataFrame()
            df = df.append(result_to_save, ignore_index=True)

        os.makedirs('lstm_res', exist_ok=True)
        df.to_csv(f'lstm_res/{save_filename}', index=False)

    return result_dict
