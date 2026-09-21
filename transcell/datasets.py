"""Dataset builders for TransCell source-target translation."""

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import scipy.sparse
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class TranslationBatchLoaders:
    train: DataLoader
    inference: DataLoader


def _obs_int_list(adata, key: str, fallback_key: Optional[str] = None) -> List[int]:
    if key in adata.obs:
        values = adata.obs[key]
    elif fallback_key and fallback_key in adata.obs:
        values = adata.obs[fallback_key]
    else:
        raise KeyError(f"Missing required obs column: {key}")
    try:
        return values.astype(int).tolist()
    except (TypeError, ValueError):
        return values.astype("category").cat.codes.astype(int).tolist()


class TranslationDataset(Dataset):
    """Sparse expression matrix plus TransCell biological annotations."""

    def __init__(
        self,
        expression,
        species_labels,
        primary_cell_type,
        cell_state,
        return_dict: bool = False,
    ):
        self.expression = expression.tocsr() if scipy.sparse.issparse(expression) else expression
        self.species_labels = list(species_labels)
        self.primary_cell_type = list(primary_cell_type)
        self.cell_state = list(cell_state)
        self.shape = expression.shape
        self.return_dict = return_dict

    def __len__(self) -> int:
        return self.expression.shape[0]

    def __getitem__(self, index: int) -> Any:
        row = self.expression[index]
        if scipy.sparse.issparse(row):
            row = row.toarray()
        x = np.asarray(row).squeeze()
        species_id = self.species_labels[index]
        primary = torch.tensor(self.primary_cell_type[index])
        state = torch.tensor(self.cell_state[index])
        if self.return_dict:
            return {
                "expression": x,
                "species_id": species_id,
                "sample_index": index,
                "cell_type": primary,
                "cell_state": state,
            }
        return x, species_id, index, primary, state


def _pad_gene_matrix(matrix, max_gene: int):
    if matrix.shape[1] >= max_gene:
        return matrix
    padding = scipy.sparse.coo_matrix(np.zeros((matrix.shape[0], max_gene - matrix.shape[1])))
    return scipy.sparse.hstack((matrix, padding))


def build_translation_dataloaders(
    adatas,
    common_adata,
    max_gene: int,
    domain_name: str = "domain_id",
    batch_size: int = 256,
    drop_last: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    use_specific: bool = True,
) -> TranslationBatchLoaders:
    """Build dataloaders for the common-gene TransCell training setting."""

    batches = common_adata.obs[domain_name].cat.categories.tolist()
    if use_specific:
        expression = None
        cell_types = []
        cell_states = []
        for index, adata in enumerate(adatas):
            adata_common = common_adata[common_adata.obs[domain_name] == batches[index],]
            common_x = adata_common.X
            specific_x = _pad_gene_matrix(adata.X, max_gene)
            block = scipy.sparse.hstack((common_x, specific_x))
            expression = block if expression is None else scipy.sparse.vstack((expression, block))
            cell_types.extend(_obs_int_list(adata, "cell_type"))
            cell_states.extend(
                _obs_int_list(adata, "transcell_cell_state", "author_cell_type_subcluster")
            )
        expression = expression.tocsr()
    else:
        expression = common_adata.X
        cell_types = _obs_int_list(common_adata, "cell_type")
        cell_states = _obs_int_list(
            common_adata, "transcell_cell_state", "author_cell_type_subcluster"
        )

    dataset = TranslationDataset(
        expression,
        common_adata.obs[domain_name].astype(int).tolist(),
        cell_types,
        cell_states,
    )
    train = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    inference = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    return TranslationBatchLoaders(train=train, inference=inference)
