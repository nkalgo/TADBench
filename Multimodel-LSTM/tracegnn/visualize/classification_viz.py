from typing import *

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

__all__ = [
    'plot_proba_cdf',
    'plot_anomaly_auc_curve',
]


def plot_proba_cdf(proba: np.ndarray,
                   truth: np.ndarray,
                   log_x: bool = True,
                   figsize: Tuple[int, int] = (16, 8),
                   labels: Optional[Dict[int, str]] = None,
                   title: Optional[str] = None,
                   output_file: Optional[str] = None
                   ):
    """Plot the CDF and KDE curve of proba (splitted by label)."""
    # inspect the data
    unique_labels = [int(i) for i in sorted(set(truth))]
    if labels is None:
        labels = {lbl: f'Label {lbl}' for lbl in unique_labels}

    # start to plot
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    if title is not None:
        fig.suptitle(title)
    for a in ax:
        a.set_xlabel('proba')
        if log_x:
            a.set_xscale('symlog')

    # CDF
    for lbl in unique_labels:
        sns.ecdfplot(proba[truth == lbl], label=labels[lbl], ax=ax[0])
    # PDF
    for lbl in unique_labels:
        sns.kdeplot(proba[truth == lbl], label=labels[lbl], ax=ax[1])

    plt.legend()
    plt.tight_layout()

    if output_file is not None:
        fig.savefig(output_file)
        plt.close()
    else:
        return fig


def plot_anomaly_auc_curve(algorithms: Dict[str, Tuple[np.ndarray, np.ndarray]],
                           figsize: Tuple[int, int] = (8, 8),
                           title: Optional[str] = None,
                           output_file: Optional[str] = None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if title is not None:
        fig.suptitle(title)

    for algorithm, (proba, truth) in algorithms.items():
        # check the data
        if set(int(v) for v in np.unique(truth)) != {0, 1}:
            raise ValueError(f'`truth` is not binary in {algorithm!r}.')

        # inspect the data
        precision, recall, thresholds = precision_recall_curve(truth, proba)

        # plot
        display = PrecisionRecallDisplay(precision, recall)
        display.plot(ax=ax, label=algorithm)

    plt.xlabel('recall')
    plt.xticks(np.linspace(0, 1, 11))
    plt.ylabel('precision')
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid(b=True, which='major')
    plt.legend()
    plt.tight_layout()

    if output_file is not None:
        fig.savefig(output_file)
        plt.close()
    else:
        return fig
