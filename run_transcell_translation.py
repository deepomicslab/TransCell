#!/usr/bin/env python3
"""
Run a TransCell human-mouse translation experiment.

Example:
    python run_transcell_translation.py --preset limb_fidelity
    python run_transcell_translation.py --pair_id pair_26 --uot_weight 400 --kl_fidelity_weight 0.7
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/transcell_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/transcell_numba")

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_PARENT = PROJECT_DIR.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

try:
    from .transcell_config import (  # noqa: E402
        DEFAULT_DATA_ROOT,
        DEFAULT_GENE_MAPPING,
        DEFAULT_OUTPUT_DIR,
        HUMAN_MOUSE_TRANSLATION_PRESETS,
    )
except ImportError:
    from transcell_config import (  # noqa: E402
        DEFAULT_DATA_ROOT,
        DEFAULT_GENE_MAPPING,
        DEFAULT_OUTPUT_DIR,
        HUMAN_MOUSE_TRANSLATION_PRESETS,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TransCell human-mouse single-cell translation."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(HUMAN_MOUSE_TRANSLATION_PRESETS),
        default="limb_fidelity",
        help="Paper-oriented parameter preset.",
    )
    parser.add_argument("--pair_id", "--pair_dir", dest="pair_id", default=None)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--gene_mapping", type=Path, default=DEFAULT_GENE_MAPPING)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--source_species", default=None)
    parser.add_argument("--target_species", default=None)
    parser.add_argument("--uot_weight", "--ot", dest="uot_weight", type=float, default=None)
    parser.add_argument(
        "--kl_fidelity_weight", "--kl", dest="kl_fidelity_weight", type=float, default=None
    )
    parser.add_argument("--reconstruction_weight", type=float, default=None)
    parser.add_argument("--hvg_count", "--gene_num", dest="hvg_count", type=int, default=None)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--ot_entropy", type=float, default=None)
    parser.add_argument("--ot_marginal_relaxation", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=124)

    return apply_preset(parser.parse_args())


def apply_preset(args):
    preset = HUMAN_MOUSE_TRANSLATION_PRESETS[args.preset]
    for key in (
        "pair_id",
        "source_species",
        "target_species",
        "uot_weight",
        "kl_fidelity_weight",
        "reconstruction_weight",
        "hvg_count",
        "iteration",
        "batch_size",
        "learning_rate",
        "ot_entropy",
        "ot_marginal_relaxation",
    ):
        if getattr(args, key) is None:
            setattr(args, key, getattr(preset, key))
    args.run_label = (
        f"{args.preset}"
        f"__uot={args.uot_weight:g}"
        f"__klfid={args.kl_fidelity_weight:g}"
        f"__hvg={args.hvg_count}"
    )
    return args


def setup_torch_path_patch():
    import torch

    original_torch_load = torch.load

    def patched_torch_load(f, *args, **kwargs):
        if isinstance(f, str) and "///" in f:
            return original_torch_load(f.replace("///", "/"), *args, **kwargs)
        return original_torch_load(f, *args, **kwargs)

    torch.load = patched_torch_load


def sort_genes_alphabetically(adata):
    return adata[:, sorted(adata.var_names)].copy()


def preprocess_for_hvg(adata):
    import scanpy as sc

    adata = adata.copy()
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def read_pair(args):
    import scanpy as sc

    pair_path = args.data_root / args.pair_id
    required = {
        "mouse_train": pair_path / "mouse_train.h5ad",
        "human_train": pair_path / "human_train.h5ad",
        "mouse_test": pair_path / "mouse_test.h5ad",
        "human_test": pair_path / "human_test.h5ad",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing pair files:\n"
            + "\n".join(missing)
            + "\nSet --data_root or --pair_id to the correct benchmark folder."
        )

    print_step("Load human-mouse pair")
    print(f"Pair: {args.pair_id}")
    return (
        sc.read_h5ad(required["mouse_train"]),
        sc.read_h5ad(required["human_train"]),
        sc.read_h5ad(required["mouse_test"]),
        sc.read_h5ad(required["human_test"]),
    )


def annotate_species_domains(adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test):
    for adata, domain_id, source in (
        (adata_mouse_train, 0, "mouse"),
        (adata_human_train, 1, "human"),
        (adata_mouse_test, -1, "mouse_query"),
        (adata_human_test, -2, "human_reference"),
    ):
        adata.obs["domain_id"] = domain_id
        adata.obs["domain_id"] = adata.obs["domain_id"].astype("category")
        adata.obs["source"] = source


def map_mouse_to_human_orthologs(adata_mouse, mapping_path):
    import pandas as pd

    mapping_df = pd.read_csv(mapping_path)
    required = {"mouse_ensembl_gene", "human_ensembl_gene"}
    if not required.issubset(mapping_df.columns):
        raise ValueError(
            f"{mapping_path} must contain columns: {', '.join(sorted(required))}"
        )

    mouse_to_human = dict(
        zip(mapping_df["mouse_ensembl_gene"], mapping_df["human_ensembl_gene"])
    )
    new_gene_names = []
    valid_indices = []
    used_human_genes = set()
    for i, gene in enumerate(adata_mouse.var_names):
        human_gene = mouse_to_human.get(gene)
        if human_gene and human_gene not in used_human_genes:
            new_gene_names.append(human_gene)
            valid_indices.append(i)
            used_human_genes.add(human_gene)

    if not valid_indices:
        raise ValueError("No mouse genes could be mapped to human ortholog IDs.")

    mapped = adata_mouse[:, valid_indices].copy()
    mapped.var_names = new_gene_names
    return mapped


def ensure_cell_type_columns(adatas):
    import pandas as pd

    for adata in adatas:
        if "cell_type" not in adata.obs:
            for candidate in ("Cell_type", "celltype", "cell_type_label", "annotation"):
                if candidate in adata.obs:
                    adata.obs["cell_type"] = adata.obs[candidate]
                    break
        if "cell_type" not in adata.obs:
            adata.obs["cell_type"] = "unknown"

        adata.obs["Cell_type"] = adata.obs["cell_type"].astype(str)
        codes = pd.Categorical(adata.obs["cell_type"]).codes
        adata.obs["cell_type"] = codes.astype(int)

        if "transcell_cell_state" in adata.obs:
            subcluster = adata.obs["transcell_cell_state"]
        elif "author_cell_type_subcluster" in adata.obs:
            subcluster = adata.obs["author_cell_type_subcluster"]
        else:
            subcluster = adata.obs["Cell_type"]
        adata.obs["transcell_cell_state"] = subcluster.astype(str)
        adata.obs["author_cell_type_subcluster"] = pd.Categorical(subcluster).codes.astype(int)


def prepare_pair(args, adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test):
    print_step("Preprocess ortholog genes")
    if not args.gene_mapping.exists():
        raise FileNotFoundError(f"Gene mapping file not found: {args.gene_mapping}")

    adatas = [
        sort_genes_alphabetically(adata)
        for adata in (adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test)
    ]
    adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test = adatas
    annotate_species_domains(adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test)

    adata_mouse_train = map_mouse_to_human_orthologs(adata_mouse_train, args.gene_mapping)
    adata_mouse_test = map_mouse_to_human_orthologs(adata_mouse_test, args.gene_mapping)
    ensure_cell_type_columns(
        [adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test]
    )
    print(f"Mouse train after ortholog mapping: {adata_mouse_train.shape}")
    print(f"Mouse test after ortholog mapping: {adata_mouse_test.shape}")
    return adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test


def select_shared_hvgs(args, adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test):
    print_step("Select shared high-variance ortholog genes")
    adata_cm = adata_mouse_train.concatenate(
        adata_human_train,
        join="inner",
        batch_key="domain_id",
    )
    ensure_cell_type_columns([adata_cm])
    adata_cm_processed = preprocess_for_hvg(adata_cm)
    sc.pp.highly_variable_genes(
        adata_cm_processed,
        n_top_genes=args.hvg_count,
        batch_key="domain_id",
        inplace=True,
        subset=True,
    )
    variable_genes = adata_cm_processed.var_names[
        adata_cm_processed.var["highly_variable"]
    ]
    adata_cm = adata_cm[:, variable_genes].copy()
    adata_mouse_train = adata_mouse_train[:, variable_genes].copy()
    adata_human_train = adata_human_train[:, variable_genes].copy()
    adata_mouse_test = adata_mouse_test[:, variable_genes].copy()
    adata_human_test = adata_human_test[:, variable_genes].copy()
    print(f"Shared HVGs: {len(variable_genes)}")
    return adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test, adata_cm


def build_cell_type_prior(mouse_cell_type, human_cell_type, alpha=2.0):
    import numpy as np
    import torch

    mouse_cell_type = np.asarray(mouse_cell_type)
    human_cell_type = np.asarray(human_cell_type)
    prior = alpha * torch.ones(len(mouse_cell_type), len(human_cell_type))
    for label in set(mouse_cell_type):
        mouse_idx = np.where(mouse_cell_type == label)[0]
        human_idx = np.where(human_cell_type == label)[0]
        if len(human_idx):
            prior[mouse_idx[:, None], human_idx] = 1 / alpha
    return prior


def train_transcell(args, adata_mouse_train, adata_human_train, adata_cm):
    from transcell.config import TranslationConfig
    from transcell.pipeline import train_cross_species_translator

    print_step("Train TransCell translation model")
    prior = build_cell_type_prior(
        adata_mouse_train.obs["cell_type"].values,
        adata_human_train.obs["cell_type"].values,
    )
    config = TranslationConfig(
        source_species=args.source_species,
        target_species=args.target_species,
        reconstruction_fidelity=args.reconstruction_weight,
        kl_fidelity=args.kl_fidelity_weight,
        ot_alignment=args.uot_weight,
        ot_entropy=args.ot_entropy,
        ot_marginal_relaxation=args.ot_marginal_relaxation,
        learning_rate=args.learning_rate,
        iteration=args.iteration,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        pair_id=args.pair_id,
        run_label=args.run_label,
        gpu=args.gpu,
        seed=args.seed,
    )
    result = train_cross_species_translator(
        adata_mouse_train,
        adata_human_train,
        adata_cm,
        config,
        prior=prior,
        save_transport=True,
    )
    return result.adata, prior, result.transport_plan


def plot_umap(adata, color, path, use_rep=None):
    import matplotlib.pyplot as plt
    import scanpy as sc

    if use_rep:
        sc.pp.neighbors(adata, use_rep=use_rep)
    else:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
    sc.tl.umap(adata, min_dist=0.1)
    sc.pl.umap(adata, color=color, frameon=False, legend_loc="right margin", show=False)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def save_translation_outputs(args, adata_pred, adata_human_test):
    import pandas as pd

    predicted = pd.DataFrame(
        adata_pred.obsm["predict"],
        index=adata_pred.obs_names,
        columns=adata_human_test.var_names,
    )
    truth = pd.DataFrame(
        adata_human_test.X.toarray() if hasattr(adata_human_test.X, "toarray") else adata_human_test.X,
        index=adata_human_test.obs_names,
        columns=adata_human_test.var_names,
    )
    predicted_path = args.output_dir / f"{args.pair_id}__predicted_human_expression__{args.run_label}.csv"
    truth_path = args.output_dir / f"{args.pair_id}__measured_human_expression__{args.run_label}.csv"
    predicted.to_csv(predicted_path)
    truth.to_csv(truth_path)
    print(f"Predicted human expression: {predicted_path}")
    print(f"Measured human expression: {truth_path}")


def predict_human_state(args, adata_mouse_test, adata_human_test, prior):
    from transcell.config import TranslationConfig
    from transcell.pipeline import predict_human_expression

    print_step("Predict human expression state from mouse query cells")
    config = TranslationConfig(
        source_species=args.source_species,
        target_species=args.target_species,
        reconstruction_fidelity=args.reconstruction_weight,
        kl_fidelity=args.kl_fidelity_weight,
        ot_alignment=args.uot_weight,
        ot_entropy=args.ot_entropy,
        ot_marginal_relaxation=args.ot_marginal_relaxation,
        learning_rate=args.learning_rate,
        iteration=args.iteration,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        pair_id=args.pair_id,
        run_label=args.run_label,
        gpu=args.gpu,
        seed=args.seed,
    )
    adata_pred = predict_human_expression(adata_mouse_test, config).adata
    save_translation_outputs(args, adata_pred, adata_human_test)
    return adata_pred


def print_step(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_torch_path_patch()

    print_step("TransCell human-mouse translation")
    print(f"Preset: {args.preset}")
    print(f"Run label: {args.run_label}")
    print(f"Pair: {args.pair_id}")
    print(f"UOT weight: {args.uot_weight}")
    print(f"KL fidelity weight: {args.kl_fidelity_weight}")
    print(f"HVG count: {args.hvg_count}")
    print(f"Output: {args.output_dir}")

    adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test = read_pair(args)
    adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test = prepare_pair(
        args, adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test
    )
    (
        adata_mouse_train,
        adata_human_train,
        adata_mouse_test,
        adata_human_test,
        adata_cm,
    ) = select_shared_hvgs(
        args, adata_mouse_train, adata_human_train, adata_mouse_test, adata_human_test
    )

    adata, prior, _ = train_transcell(args, adata_mouse_train, adata_human_train, adata_cm)

    integrated_path = args.output_dir / f"{args.pair_id}__integrated_latent__{args.run_label}.png"
    baseline_path = args.output_dir / f"{args.pair_id}__pre_transcell_pca__{args.run_label}.png"
    plot_umap(adata, ["source", "Cell_type"], integrated_path, use_rep="latent")
    plot_umap(adata_cm.copy(), ["source", "Cell_type"], baseline_path)
    print(f"Integrated latent UMAP: {integrated_path}")
    print(f"Pre-TransCell PCA UMAP: {baseline_path}")

    adata_pred = predict_human_state(args, adata_mouse_test, adata_human_test, prior)
    prediction_path = args.output_dir / f"{args.pair_id}__predicted_human_umap__{args.run_label}.png"
    plot_umap(adata_pred, ["Cell_type"], prediction_path, use_rep="predict")
    print(f"Predicted human-state UMAP: {prediction_path}")


if __name__ == "__main__":
    main()
