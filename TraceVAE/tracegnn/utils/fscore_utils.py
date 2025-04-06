from typing import *

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

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


def best_fscore(proba: np.ndarray,
                truth: np.ndarray) -> Tuple[float, float]:
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
    precision = round(TP / (TP + FP), 6) if (TP + FP) != 0 else 0
    recall = round(TP / (TP + FN), 6) if (TP + FN) != 0 else 0
    F1_score = round(2 * precision * recall / (precision + recall), 6) if (precision + recall) != 0 else 0
    ACC = round((TP + TN) / (TP + FN + FP + TN), 6) if (TP + FN + FP + TN) != 0 else 0
    return F1_score, threshold, precision, recall, TP, TN, FN, FP, precision, recall, F1_score, ACC

def auc_score(proba: np.ndarray, truth: np.ndarray) -> float:
    return float(average_precision_score(truth, proba))
