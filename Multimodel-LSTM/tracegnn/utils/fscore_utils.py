from typing import *

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

__all__ = [
    'fscore_for_precision_and_recall',
    'best_fscore',
    'auc_score',
]


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


def new_fscore(proba: np.ndarray, truth: np.ndarray):
    truth = truth.tolist()
    TP, FP, TN, FN = 0, 0, 0, 0
    scores = sorted(proba, reverse=True)
    threshold = scores[len(scores) // 100 * 15]  # 选择排在前 15% 的得分作为阈值
    predict = []
    for p in proba:
        predict.append(1 if p > threshold else 0)
    for i in range(len(predict)):
        if truth[i] is True and predict[i] == 1:
            TP += 1
        elif truth[i] is False and predict[i] == 0:
            TN += 1
        elif truth[i] is False and predict[i] == 1:
            FP += 1
        else:
            FN += 1
    precision = round(TP / (TP + FP), 6)
    recall = round(TP / (TP + FN), 6)
    F1_score = round(2 * precision * recall / (precision + recall), 6)
    ACC = round((TP + TN) / (TP + FN + FP + TN), 6)
    return F1_score, threshold, precision, recall, TP, TN, FN, FP, precision, recall, F1_score, ACC


def best_fscore_with_counts(proba: np.ndarray, truth: np.ndarray) -> Tuple[float, float, int, int, int, int]:
    precision, recall, thresholds = precision_recall_curve(truth, proba)
    fscore = 2 * (precision * recall) / (precision + recall)
    # 防止除以零的错误
    fscore[np.isnan(fscore)] = 0

    # 找到最大化F-score的索引（注意跳过最后一个点，因为precision和recall中可能包含-inf）
    idx = np.argmax(fscore[:-1])
    best_threshold = thresholds[idx]

    # 使用最佳阈值将概率转化为预测标签
    predicted_labels = (proba >= best_threshold).astype(int)

    # 计算混淆矩阵
    cm = confusion_matrix(truth, predicted_labels)

    # 从混淆矩阵中提取TP, TN, FP, FN
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # 返回F-score, 阈值, TP, TN, FP, FN
    return fscore[idx], best_threshold, TP, TN, FP, FN


def best_fscore(proba: np.ndarray,
                truth: np.ndarray) -> Tuple[float, float]:
    float32_max = np.finfo(np.float32).max
    float32_min = np.finfo(np.float32).min
    if np.isnan(proba).any():
        proba = np.nan_to_num(proba)
        print("proba数据包含 NaN")
    elif np.isinf(proba).any():
        print("proba数据包含 Infinity 或 -Infinity")
    elif (proba > float32_max).any() or (proba < float32_min).any():
        print("proba数据包含超出 float32 范围的值")

    if np.isnan(truth).any():
        print("truth数据包含 NaN")
    elif np.isinf(truth).any():
        print("truth数据包含 Infinity 或 -Infinity")
    elif (truth > float32_max).any() or (truth < float32_min).any():
        print("truth数据包含超出 float32 范围的值")

    precision, recall, threshold = precision_recall_curve(truth, proba)
    fscore = fscore_for_precision_and_recall(precision, recall)
    idx = np.argmax(fscore[:-1])
    best_threshold = threshold[idx]

    predictions = (proba >= best_threshold).astype(int)
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
