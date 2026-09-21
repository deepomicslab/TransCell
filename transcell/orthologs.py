"""Ortholog gene-space projection for cross-species translation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd
from anndata import AnnData


@dataclass
class OrthologProjector:
    """Project source-species genes into the target-species ortholog space."""

    source_gene_column: str = "mouse_ensembl_gene"
    target_gene_column: str = "human_ensembl_gene"

    @classmethod
    def from_csv(
        cls,
        mapping_path: Union[str, Path],
        source_gene_column: str = "mouse_ensembl_gene",
        target_gene_column: str = "human_ensembl_gene",
    ) -> "OrthologProjector":
        projector = cls(source_gene_column, target_gene_column)
        projector.fit(mapping_path)
        return projector

    def fit(self, mapping_path: Union[str, Path]) -> "OrthologProjector":
        mapping_df = pd.read_csv(mapping_path)
        required = {self.source_gene_column, self.target_gene_column}
        if not required.issubset(mapping_df.columns):
            raise ValueError(
                f"{mapping_path} must contain columns: {', '.join(sorted(required))}"
            )
        self.source_to_target_ = dict(
            zip(mapping_df[self.source_gene_column], mapping_df[self.target_gene_column])
        )
        return self

    def transform_source_to_target_gene_space(self, adata: AnnData) -> AnnData:
        if not hasattr(self, "source_to_target_"):
            raise RuntimeError("Call fit() before transforming an AnnData object.")

        target_names = []
        source_indices = []
        used_targets = set()
        for index, gene in enumerate(adata.var_names):
            target_gene = self.source_to_target_.get(gene)
            if target_gene and target_gene not in used_targets:
                source_indices.append(index)
                target_names.append(target_gene)
                used_targets.add(target_gene)

        if not source_indices:
            raise ValueError("No source genes could be mapped to target ortholog IDs.")

        projected = adata[:, source_indices].copy()
        projected.var_names = target_names
        return projected
