# TransCell Code

TransCell is an ortholog-aware cross-species single-cell expression translation
framework. The code is organized around the paper workflow:

1. map source-species genes into the target ortholog gene space;
2. select shared high-variance ortholog genes;
3. train a ZINB expression translator with latent KL fidelity, UOT alignment,
   and hierarchical cell-state constraints;
4. predict target-species expression states from source query cells;
5. evaluate expression fidelity and biological consistency.

## Package Layout

```text
transcell/
  config.py                 TranslationConfig for paper-level hyperparameters
  orthologs.py              OrthologProjector for source-to-target gene mapping
  preprocessing.py          SpeciesPair, domain annotation, shared HVG selection
  datasets.py               TranslationDataset and dataloader builders
  pipeline.py               train/predict/translate source-target workflows
  model/
    nn_blocks.py            species-aware encoder and ZINB decoder blocks
    transcell_model.py      standalone TransCellModel interface
  training/
    objectives.py           ZINB, KL, UOT, hierarchy objectives
    trainer.py              TransCell h-mode source-target training loop
    checkpoint.py           TranslationEarlyStopping
  evaluation/
    fidelity.py             expression-fidelity metrics
    biological_consistency.py
```

The paper-facing code path now uses `transcell.pipeline`,
`transcell.model.TransCellModel`, and `transcell.training.TransCellTrainer`.
Legacy UniPort-style modules have been isolated under `legacy/`. The top-level
`function.py` only provides a deprecated wrapper for older scripts and delegates
to the new TransCell pipeline.

## Main Entry

```bash
python run_transcell_translation.py --preset limb_fidelity
```

Important parameters:

- `--preset`: named paper-oriented configuration from `transcell_config.py`
- `--pair_id`: dataset pair folder, such as the folder used by a selected preset
- `--uot_weight`: unbalanced optimal transport fidelity weight
- `--kl_fidelity_weight`: latent KL fidelity regularization weight
- `--hvg_count`: number of highly variable ortholog genes
- `--data_root`: folder containing pair subfolders with `mouse_train.h5ad`,
  `human_train.h5ad`, `mouse_test.h5ad`, and `human_test.h5ad`
- `--gene_mapping`: CSV with `mouse_ensembl_gene` and `human_ensembl_gene`

## Third-Party Attribution

Core model components are adapted from UniPort/SCALEX-style code. The required
third-party notice is kept in `NOTICE`. The TransCell-specific code path centers
on ortholog projection, cross-species translation datasets, ZINB fidelity,
hierarchical cell-state constraints, UOT alignment, and fidelity evaluation.
