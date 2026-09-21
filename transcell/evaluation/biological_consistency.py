"""Biological consistency checks for TransCell outputs."""

import numpy as np


def marker_gene_preservation(predicted_df, measured_df, marker_genes):
    """Compare marker-gene pseudobulk profiles for selected genes."""

    genes = [gene for gene in marker_genes if gene in predicted_df.columns and gene in measured_df.columns]
    if not genes:
        return np.nan
    pred = predicted_df[genes].mean(axis=0).values
    truth = measured_df[genes].mean(axis=0).values
    if np.std(pred) == 0 or np.std(truth) == 0:
        return np.nan
    return float(np.corrcoef(pred, truth)[0, 1])

