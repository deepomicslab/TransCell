"""Fidelity metrics for predicted target-species expression."""

import numpy as np


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def genewise_correlation(predicted, measured):
    """Pearson correlation per gene between predicted and measured expression."""

    pred = _dense(predicted)
    truth = _dense(measured)
    corrs = []
    for gene_idx in range(pred.shape[1]):
        x = pred[:, gene_idx]
        y = truth[:, gene_idx]
        if np.std(x) == 0 or np.std(y) == 0:
            corrs.append(np.nan)
        else:
            corrs.append(np.corrcoef(x, y)[0, 1])
    return np.asarray(corrs)


def pseudobulk_correlation(predicted, measured):
    """Correlation of average predicted and measured target expression."""

    pred_mean = _dense(predicted).mean(axis=0)
    truth_mean = _dense(measured).mean(axis=0)
    if np.std(pred_mean) == 0 or np.std(truth_mean) == 0:
        return np.nan
    return float(np.corrcoef(pred_mean, truth_mean)[0, 1])

