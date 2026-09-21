"""Preprocessing primitives for TransCell species-pair workflows."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import scanpy as sc
from anndata import AnnData


@dataclass
class SpeciesPair:
    """A source-target training and evaluation bundle."""

    source_train: AnnData
    target_train: AnnData
    source_query: Optional[AnnData] = None
    target_reference: Optional[AnnData] = None
    source_species: str = "mouse"
    target_species: str = "human"
    ortholog_genes: Optional[List[str]] = None


def sort_genes_alphabetically(adata: AnnData) -> AnnData:
    return adata[:, sorted(adata.var_names)].copy()


def annotate_species_domains(
    source_train: AnnData,
    target_train: AnnData,
    source_query: Optional[AnnData] = None,
    target_reference: Optional[AnnData] = None,
    source_species: str = "mouse",
    target_species: str = "human",
) -> None:
    records = [
        (source_train, 0, source_species),
        (target_train, 1, target_species),
    ]
    if source_query is not None:
        records.append((source_query, -1, f"{source_species}_query"))
    if target_reference is not None:
        records.append((target_reference, -2, f"{target_species}_reference"))

    for adata, domain_id, source_name in records:
        adata.obs["domain_id"] = domain_id
        adata.obs["domain_id"] = adata.obs["domain_id"].astype("category")
        adata.obs["source"] = source_name


def ensure_cell_state_columns(adatas: list[AnnData]) -> None:
    """Normalize cell-type columns required by hierarchy-aware training."""

    for adata in adatas:
        if "cell_type" not in adata.obs:
            for candidate in ("Cell_type", "celltype", "cell_type_label", "annotation"):
                if candidate in adata.obs:
                    adata.obs["cell_type"] = adata.obs[candidate]
                    break
        if "cell_type" not in adata.obs:
            adata.obs["cell_type"] = "unknown"

        adata.obs["Cell_type"] = adata.obs["cell_type"].astype(str)
        adata.obs["cell_type"] = pd.Categorical(adata.obs["cell_type"]).codes.astype(int)

        if "transcell_cell_state" in adata.obs:
            substate = adata.obs["transcell_cell_state"]
        elif "author_cell_type_subcluster" in adata.obs:
            substate = adata.obs["author_cell_type_subcluster"]
        else:
            substate = adata.obs["Cell_type"]

        adata.obs["transcell_cell_state"] = substate.astype(str)
        adata.obs["author_cell_type_subcluster"] = pd.Categorical(substate).codes.astype(int)


def preprocess_for_hvg(adata: AnnData) -> AnnData:
    processed = adata.copy()
    if "counts" not in processed.layers:
        processed.layers["counts"] = processed.X.copy()
    sc.pp.normalize_total(processed, target_sum=1e4)
    sc.pp.log1p(processed)
    return processed


def select_shared_ortholog_hvgs(
    source_train: AnnData,
    target_train: AnnData,
    hvg_count: int,
    source_query: Optional[AnnData] = None,
    target_reference: Optional[AnnData] = None,
) -> Tuple[AnnData, AnnData, Optional[AnnData], Optional[AnnData], AnnData]:
    """Select shared high-variance genes and return an AnnData common space."""

    common = source_train.concatenate(target_train, join="inner", batch_key="domain_id")
    ensure_cell_state_columns([common])
    processed = preprocess_for_hvg(common)
    sc.pp.highly_variable_genes(
        processed,
        n_top_genes=hvg_count,
        batch_key="domain_id",
        inplace=True,
        subset=True,
    )
    genes = processed.var_names[processed.var["highly_variable"]]
    source_train = source_train[:, genes].copy()
    target_train = target_train[:, genes].copy()
    common = common[:, genes].copy()
    if source_query is not None:
        source_query = source_query[:, genes].copy()
    if target_reference is not None:
        target_reference = target_reference[:, genes].copy()
    return source_train, target_train, source_query, target_reference, common
