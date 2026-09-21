#!/usr/bin/env python3
"""Deprecated compatibility helpers for older TransCell scripts.

New code should import from ``transcell``.
"""

import os
import warnings
from glob import glob

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import sklearn
import torch
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.preprocessing import MaxAbsScaler

from .transcell.config import TranslationConfig
from .transcell.io import read_single_cell_file
from .transcell.pipeline import (
    predict_human_expression,
    train_cross_species_translator,
)


DATA_PATH = os.path.expanduser("~") + "/.transcell/"
CHUNK_SIZE = 20000


def read_mtx(path):
    adata = None
    for filename in glob(path + "/*"):
        if ("count" in filename or "matrix" in filename or "data" in filename) and "mtx" in filename:
            adata = sc.read_mtx(filename).T
    if adata is None:
        raise ValueError(f"No matrix file found in {path}")
    for filename in glob(path + "/*"):
        if "barcode" in filename:
            barcode = pd.read_csv(filename, sep="\t", header=None).iloc[:, -1].values
            adata.obs = pd.DataFrame(index=barcode)
        if "gene" in filename or "peaks" in filename:
            gene = pd.read_csv(filename, sep="\t", header=None).iloc[:, -1].values
            adata.var = pd.DataFrame(index=gene)
        elif "feature" in filename:
            gene = pd.read_csv(filename, sep="\t", header=None).iloc[:, 1].values
            adata.var = pd.DataFrame(index=gene)
    return adata


def load_file(path):
    if os.path.exists(DATA_PATH + path + ".h5ad"):
        adata = sc.read_h5ad(DATA_PATH + path + ".h5ad")
    elif os.path.isdir(path):
        adata = read_mtx(path)
    elif os.path.isfile(path):
        adata = read_single_cell_file(path)
    else:
        raise ValueError(f"File {path} does not exist")
    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    return adata


def tfidf(X, n_components, binarize=True, random_state=0):
    from sklearn.feature_extraction.text import TfidfTransformer

    sc_count = np.copy(X)
    if binarize:
        sc_count = np.where(sc_count < 1, sc_count, 1)
    normed_count = TfidfTransformer(norm="l2", sublinear_tf=True).fit_transform(sc_count)
    lsi = sklearn.decomposition.TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
    )
    return lsi.fit_transform(normed_count)[:, 1:]


def TFIDF_LSI(adata, n_comps=50, binarize=True, random_state=0):
    if scipy.sparse.issparse(adata.X):
        adata.obsm["X_lsi"] = tfidf(
            adata.X.todense(),
            n_components=n_comps,
            binarize=binarize,
            random_state=random_state,
        )
    else:
        adata.obsm["X_lsi"] = tfidf(
            adata.X,
            n_components=n_comps,
            binarize=binarize,
            random_state=random_state,
        )


def filter_data(adata: AnnData, min_features: int = 0, min_cells: int = 0, log=None):
    if log:
        log.info("Filtering cells")
    sc.pp.filter_cells(adata, min_genes=min_features)
    if log:
        log.info("Filtering features")
    sc.pp.filter_genes(adata, min_cells=min_cells)


def batch_scale(adata, use_rep="X", chunk_size=CHUNK_SIZE):
    for batch in adata.obs["source"].unique():
        idx = np.where(adata.obs["source"] == batch)[0]
        if use_rep == "X":
            scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
            for i in range(len(idx) // chunk_size + 1):
                adata.X[idx[i * chunk_size : (i + 1) * chunk_size]] = scaler.transform(
                    adata.X[idx[i * chunk_size : (i + 1) * chunk_size]]
                )
        else:
            scaler = MaxAbsScaler(copy=False).fit(adata.obsm[use_rep][idx])
            for i in range(len(idx) // chunk_size + 1):
                adata.obsm[use_rep][idx[i * chunk_size : (i + 1) * chunk_size]] = scaler.transform(
                    adata.obsm[use_rep][idx[i * chunk_size : (i + 1) * chunk_size]]
                )


def get_prior(celltype1, celltype2, alpha=2):
    couple = alpha * torch.ones(len(celltype1), len(celltype2))
    for label in set(celltype1):
        index1 = np.where(celltype1 == label)
        if label in set(celltype2):
            index2 = np.where(celltype2 == label)
            for row in index1[0]:
                couple[row, index2[0]] = 1 / alpha
    return couple


def label_reweight(celltype):
    n_obs = len(celltype)
    unique, count = np.unique(celltype, return_counts=True)
    weights = torch.zeros(n_obs, 1)
    for i in range(n_obs):
        idx = np.where(unique == celltype[i])[0]
        weights[i] = torch.from_numpy(1 / (len(unique) * count[idx]))
    return weights * n_obs


def _legacy_config(
    lambda_s,
    lambda_recon,
    lambda_kl,
    lambda_ot,
    iteration,
    reg,
    reg_m,
    batch_size,
    lr,
    gpu,
    seed,
    num_workers,
    patience,
    outdir,
    pair,
    super_hyp,
):
    return TranslationConfig(
        species_specific_reconstruction=lambda_s,
        reconstruction_fidelity=lambda_recon,
        kl_fidelity=lambda_kl,
        ot_alignment=lambda_ot,
        iteration=iteration,
        ot_entropy=reg,
        ot_marginal_relaxation=reg_m,
        batch_size=batch_size,
        learning_rate=lr,
        gpu=gpu,
        seed=seed,
        num_workers=num_workers,
        patience=patience,
        output_dir=outdir,
        pair_id=pair or "legacy_pair",
        run_label=super_hyp or "legacy_run",
    )


def _deprecated_run(
    adatas=None,
    adata_cm=None,
    mode="h",
    lambda_s=0.5,
    lambda_recon=1.0,
    lambda_kl=5.0,
    lambda_ot=1.0,
    iteration=30000,
    ref_id=None,
    save_OT=False,
    use_rep=None,
    out="latent",
    label_weight=None,
    reg=0.1,
    reg_m=1.0,
    batch_size=256,
    lr=2e-4,
    enc=None,
    gpu=0,
    prior=None,
    loss_type="BCE",
    outdir="output",
    input_id=0,
    pred_id=1,
    seed=124,
    num_workers=4,
    patience=30,
    batch_key="domain_id",
    source_name="source",
    model_info=False,
    verbose=False,
    pair=None,
    super_hyp=None,
):
    warnings.warn(
        "Run is deprecated; use transcell.pipeline instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if mode != "h":
        raise NotImplementedError("The compatibility wrapper only supports mode='h'.")
    config = _legacy_config(
        lambda_s,
        lambda_recon,
        lambda_kl,
        lambda_ot,
        iteration,
        reg,
        reg_m,
        batch_size,
        lr,
        gpu,
        seed,
        num_workers,
        patience,
        outdir,
        pair,
        super_hyp,
    )
    if out == "latent":
        if adatas is None or len(adatas) != 2 or adata_cm is None:
            raise ValueError("Legacy latent mode requires two adatas and adata_cm.")
        result = train_cross_species_translator(
            adatas[0],
            adatas[1],
            adata_cm,
            config,
            prior=prior,
            label_weight=label_weight,
            save_transport=save_OT,
            batch_key=batch_key,
            source_name=source_name,
            model_info=model_info,
            verbose=verbose,
        )
        if save_OT:
            return result.adata, result.transport_plan
        return result.adata
    if out == "predict":
        if adata_cm is None:
            raise ValueError("Legacy predict mode requires adata_cm as query AnnData.")
        return predict_human_expression(
            adata_cm,
            config,
            pred_id=pred_id,
            batch_key=batch_key,
        ).adata
    raise ValueError("Unsupported legacy out mode: {}".format(out))


Run = _deprecated_run
