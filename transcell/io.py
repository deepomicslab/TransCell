"""Input helpers for TransCell datasets."""

from pathlib import Path
from typing import Union

import pandas as pd
import scanpy as sc
import scipy.sparse
from anndata import AnnData


def read_single_cell_file(path: Union[str, Path]) -> AnnData:
    """Read a supported single-cell matrix into sparse AnnData."""

    path = Path(path)
    if path.suffix == ".h5ad":
        adata = sc.read_h5ad(path)
    elif path.suffix in {".csv", ".gz"} and path.name.endswith((".csv", ".csv.gz")):
        adata = sc.read_csv(str(path)).T
    elif path.name.endswith((".txt", ".txt.gz", ".tsv", ".tsv.gz")):
        df = pd.read_csv(path, sep="\t", index_col=0).T
        adata = AnnData(
            df.values,
            obs=pd.DataFrame(index=df.index.values),
            var=pd.DataFrame(index=df.columns.values),
        )
    else:
        raise ValueError(f"Unsupported single-cell file: {path}")

    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    return adata
