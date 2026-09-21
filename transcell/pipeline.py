"""Paper-oriented TransCell translation pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from logger import create_logger

from .config import TranslationConfig
from .datasets import build_translation_dataloaders
from .model.transcell_model import TransCellModel
from .training.checkpoint import TranslationEarlyStopping
from .training.trainer import TransCellTrainer


@dataclass
class TranslationRunResult:
    """Outputs from a TransCell translation run."""

    adata: object
    transport_plan: Optional[Dict] = None
    model: Optional[TransCellModel] = None


def build_cell_type_prior(source_cell_type, target_cell_type, alpha=2.0):
    """Create a cell-type-informed correspondence prior."""

    source_cell_type = np.asarray(source_cell_type)
    target_cell_type = np.asarray(target_cell_type)
    prior = alpha * torch.ones(len(source_cell_type), len(target_cell_type))
    for label in set(source_cell_type):
        source_idx = np.where(source_cell_type == label)[0]
        target_idx = np.where(target_cell_type == label)[0]
        if len(target_idx):
            prior[source_idx[:, None], target_idx] = 1 / alpha
    return prior


def _device_from_config(config: TranslationConfig):
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
        return "cuda"
    return "cpu"


def _checkpoint_dir(config: TranslationConfig) -> Path:
    return Path(config.output_dir) / "checkpoint" / config.pair_id / config.run_label


def _num_cells_and_genes(adatas, use_rep):
    num_cell = []
    num_gene = []
    for index, adata in enumerate(adatas):
        if use_rep[index] == "X":
            num_cell.append(adata.X.shape[0])
            num_gene.append(adata.X.shape[1])
        else:
            num_cell.append(adata.obsm[use_rep[index]].shape[0])
            num_gene.append(adata.obsm[use_rep[index]].shape[1])
    return num_cell, num_gene


def _default_architecture(num_gene, common_gene_count, n_domain):
    enc = [
        ["fc", 1024, 1, "leaky_relu", 0.1],
        ["fc", 1024, 1, "leaky_relu", 0.1],
        ["fc", 1024, 1, "leaky_relu", 0.1],
        ["fc", 100, 1, "", 0],
    ]
    dec = {
        0: [
            ["fc", 256, n_domain, "leaky_relu", 0],
            ["fc", 256, n_domain, "leaky_relu", 0],
            ["fc", 256, n_domain, "leaky_relu", 0],
            ["fc", 256, n_domain, "relu", 0],
            ["", common_gene_count, "", "", 0],
        ]
    }
    for index in range(1, n_domain + 1):
        dec[index] = [
            ["fc", 256, 1, "leaky_relu", 0],
            ["fc", 256, 1, "leaky_relu", 0],
            ["fc", 256, 1, "leaky_relu", 0],
            ["fc", 256, 1, "relu", 0],
            ["", num_gene[index - 1], "", "", 0],
        ]
    return enc, dec


def train_cross_species_translator(
    source_train,
    target_train,
    common_adata,
    config: TranslationConfig,
    prior=None,
    label_weight=None,
    save_transport=True,
    batch_key="domain_id",
    source_name="source",
    model_info=False,
    verbose=False,
):
    """Train TransCell in the paper's source-to-target translation setting."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _device_from_config(config)
    checkpoint_dir = _checkpoint_dir(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log = create_logger("", fh=str(checkpoint_dir / "log.txt"))

    adatas = [source_train, target_train]
    n_domain = len(adatas)
    ref_id = n_domain - 1
    num_cell, num_gene = _num_cells_and_genes(adatas, ["X", "X"])
    common_gene_count = common_adata.X.shape[1]

    transport_plan = {}
    if save_transport:
        for domain_id in range(n_domain):
            if domain_id != ref_id:
                ns = num_cell[domain_id]
                nt = num_cell[ref_id]
                transport_plan[domain_id] = (
                    np.ones((ns, nt), dtype=np.float32) / (ns * nt)
                ).astype(np.float32)

    loaders = build_translation_dataloaders(
        adatas=adatas,
        common_adata=common_adata,
        max_gene=max(num_gene),
        domain_name=batch_key,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_specific=True,
    )

    num_gene_with_common = list(num_gene) + [common_gene_count]
    enc, dec = _default_architecture(num_gene, common_gene_count, n_domain)
    model = TransCellModel(
        enc,
        dec,
        reference_species_id=ref_id,
        n_species=n_domain,
    )
    if model_info:
        log.info("model\n" + model.__repr__())

    early_stopping = TranslationEarlyStopping(
        patience=config.patience,
        checkpoint_file=str(checkpoint_dir / "model.pt"),
    )
    trainer = TransCellTrainer(model=model, config=config)
    trainer.fit(
        loaders.train,
        transport_plan,
        num_cell,
        num_gene_with_common,
        prior=prior,
        label_weight=label_weight,
        early_stopping=early_stopping,
        device=device,
        save_transport=save_transport,
        verbose=verbose,
        max_gene=max(num_gene_with_common),
    )

    torch.save(
        {
            "enc": enc,
            "dec": dec,
            "n_domain": n_domain,
            "ref_id": ref_id,
            "num_gene": num_gene_with_common,
        },
        checkpoint_dir / "config.pt",
    )
    with torch.no_grad():
        common_adata.obsm["latent"] = model.embed(
            loaders.inference,
            num_gene_with_common,
            device=device,
            eval_mode=False,
        )
    if save_transport:
        return TranslationRunResult(common_adata, transport_plan, model)
    return TranslationRunResult(common_adata, None, model)


def predict_human_expression(
    source_query,
    config: TranslationConfig,
    pred_id=1,
    batch_key="domain_id",
):
    """Predict target-species expression from source query cells."""

    device = _device_from_config(config)
    checkpoint_dir = _checkpoint_dir(config)
    state = torch.load(checkpoint_dir / "config.pt")
    enc = state["enc"]
    dec = state["dec"]
    n_domain = state["n_domain"]
    ref_id = state["ref_id"]
    num_gene = state["num_gene"]
    model = TransCellModel(
        enc,
        dec,
        reference_species_id=ref_id,
        n_species=n_domain,
    )
    model.load_model(str(checkpoint_dir / "model.pt"))
    model = model.to(device)

    query_common = source_query.copy()
    query_common.obs[batch_key] = 0
    query_common.obs[batch_key] = query_common.obs[batch_key].astype("category")
    loaders = build_translation_dataloaders(
        adatas=[query_common],
        common_adata=query_common,
        max_gene=max(num_gene),
        domain_name=batch_key,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=config.num_workers,
        use_specific=False,
    )
    with torch.no_grad():
        source_query.obsm["predict"] = model.predict_target_expression(
            loaders.inference,
            num_gene,
            target_decoder_id=pred_id,
            device=device,
            eval_mode=True,
        )
    return TranslationRunResult(source_query, None, model)


def translate_species_pair(
    source_train,
    target_train,
    common_adata,
    config: TranslationConfig,
    source_query=None,
    prior=None,
    save_transport=True,
    **kwargs,
):
    """Train on a source-target pair and optionally translate source query cells."""

    train_result = train_cross_species_translator(
        source_train,
        target_train,
        common_adata,
        config,
        prior=prior,
        save_transport=save_transport,
        **kwargs,
    )
    if source_query is None:
        return train_result
    return predict_human_expression(source_query, config)
