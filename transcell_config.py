"""
Project-level defaults for TransCell human-mouse translation experiments.
"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_DIR / "data" / "human_mouse_pairs"
DEFAULT_GENE_MAPPING = PROJECT_DIR / "data" / "orthologs" / "mouse_to_human_ensembl.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "translation_fidelity"


@dataclass(frozen=True)
class TranslationPreset:
    """Named parameter set for a TransCell translation benchmark."""

    name: str
    pair_id: str
    source_species: str = "mouse"
    target_species: str = "human"
    uot_weight: float = 400.0
    kl_fidelity_weight: float = 0.7
    reconstruction_weight: float = 1.0
    hvg_count: int = 2000
    iteration: int = 3000
    batch_size: int = 1024
    learning_rate: float = 1e-3
    ot_entropy: float = 0.1
    ot_marginal_relaxation: float = 1.0

    @property
    def run_label(self) -> str:
        return (
            f"{self.name}"
            f"__uot={self.uot_weight:g}"
            f"__klfid={self.kl_fidelity_weight:g}"
            f"__hvg={self.hvg_count}"
        )


HUMAN_MOUSE_TRANSLATION_PRESETS = {
    "limb_fidelity": TranslationPreset(
        name="limb_fidelity",
        pair_id="pair_26",
    ),
    "kidney_atlas": TranslationPreset(
        name="kidney_atlas",
        pair_id="kidney_human_mouse",
        iteration=5000,
    ),
    "muscle_aging": TranslationPreset(
        name="muscle_aging",
        pair_id="muscle_aging_human_mouse",
        hvg_count=2500,
        iteration=5000,
    ),
    "t1d_islet": TranslationPreset(
        name="t1d_islet",
        pair_id="t1d_islet_human_mouse",
        hvg_count=2500,
        iteration=5000,
    ),
}

