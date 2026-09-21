"""Configuration objects for TransCell translation workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class TranslationConfig:
    """Hyperparameters for ortholog-aware source-to-target translation."""

    source_species: str = "mouse"
    target_species: str = "human"
    reconstruction_fidelity: float = 1.0
    kl_fidelity: float = 5.0
    ot_alignment: float = 1.0
    species_specific_reconstruction: float = 0.5
    ot_entropy: float = 0.1
    ot_marginal_relaxation: float = 1.0
    learning_rate: float = 2e-4
    iteration: int = 30000
    batch_size: int = 256
    patience: int = 30
    seed: int = 124
    gpu: int = 0
    num_workers: int = 4
    output_dir: Union[Path, str] = "output"
    pair_id: str = "species_pair"
    run_label: str = "default"
