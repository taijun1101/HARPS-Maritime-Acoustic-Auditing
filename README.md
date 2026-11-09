# HARPS: Asymmetric Reference Learning for Noise-Aware Maritime Acoustic Auditing

Research reproduction code for the IEEE BigData 2026 paper **HARPS: Asymmetric Reference Learning for Noise-Aware Maritime Acoustic Auditing**.

> **Reproduction status:** The experiment has been split into modules; every original top-level function/class AST is checked against the preserved reference source. The downloaders and auxiliary checks are newly packaged. Numerical equivalence has **not** yet been established by rerunning the full grid. Do not interpret a successful smoke test as validation of the paper's results.

## Repository contents

- `run_experiments.py` — minimal CLI entry point.
- `harps/` — modular implementation: config, dataset, noise, features, metrics, baselines, deep_svdd, experiments, reporting, benchmarking, and cli.
- `reference/original_run_experiments.py` — unmodified original implementation for structural regression testing.
- `tests/test_modular_equivalence.py` — checks exact AST equality for all original top-level functions and classes.
- `scripts/download_ast.py` — downloads `MIT/ast-finetuned-audioset-10-10-0.4593`.
- `scripts/download_shipsear.py` — downloads `peng7554/DS3500` and optionally extracts `ShipsEar.zip`.
- `scripts/check_dataset.py` — checks the directory convention required by the experiment runner.
- `scripts/check_environment.py` — checks installed versions against the supplied server snapshot.
- `environment/` — original `environment.json`, `pip_freeze.txt`, `conda_explicit.txt`.
- `tests/` — syntax and packaging smoke tests.

## Environment

The original run used **Python 3.10.21**, **PyTorch 2.5.1+cu121**, **Transformers 4.46.3**, **PyOD 3.6.6**, and an NVIDIA A100 80GB PCIe GPU. See `environment/environment.json` for the recorded setup.

```bash
conda create -n ast_deep_svdd python=3.10.21 -y
conda activate ast_deep_svdd
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

If dependency resolution fails, refer to `environment/pip_freeze.txt` and `environment/conda_explicit.txt` for the source environment; platform compatibility and upstream wheel availability may vary. `requirements.txt` includes CUDA-specific wheels and is primarily intended for a compatible Linux/CUDA environment.

## Download model and dataset

```bash
python scripts/download_ast.py --output_dir ./ast_weight
python scripts/download_shipsear.py --output_dir ./data/DS3500 --extract
```

The downloaders **do not** run AST inference or preprocess audio. The downloaded `ShipsEar.zip` is **not automatically equivalent** to the original experiment's `shipsear_5s_16k` directory. The original run expects `data_root/group/class_name/file.wav`, reads `.wav` files recursively and infers the class from the second path component. Ensure the data preparation, audio segments and labels match the original experiment before attempting reproduction. The repository does not yet include a verified transformation from the downloaded archive to the author's experiment-ready dataset.

```bash
python scripts/check_dataset.py --data_root /path/to/shipsear_5s_16k
```

## Full experiment reproduction

Run from the repository root with the **same experiment-ready data and model checkpoint** used for the paper:

```bash
python run_experiments.py \
  --data_root /path/to/shipsear_5s_16k \
  --ast_model_path ./ast_weight \
  --output_dir ./results \
  --run_all \
  --run_asymmetric \
  --run_mfcc_svdd \
  --run_pink \
  --run_sensitivity \
  --run_eigen \
  --benchmark_cost \
  --batch_size 16 \
  --run_geometry_comparison
```

These are the author's supplied full-experiment flags. Unspecified settings use the original runner's defaults, including seeds `0 1 2 3 4`, evaluation SNRs `20 10 5` dB, 30 epochs, and 12 LOCO classes. This protocol is **reference-inclusive LOCO**, **not** an 80/20 split: for each held-out class, all other classes form the normal reference set, and the full dataset is scored during evaluation. It should not be described as an unseen normal holdout.

The frozen AST representation uses the **mean of the first two special-token embeddings**. The AST-only projection is a single bias-free linear layer, 768→128. The original script also implements MFCC and additional ablation configurations.

### Outputs

Results are written under `--output_dir`, including `dataset_manifest.csv`, `loco_sample_counts.csv`, `ast_metadata.json`, `experiment_config.json`, `baselines/`, `deep_svdd/`, `sensitivity/`, `feature_cache/`, and `summary/`. The `summary/` subdirectories include `overall_metrics_across_loco.csv` and other aggregation outputs. Large outputs are ignored by Git.

## Attribution and Redistribution

This repository does not redistribute ShipsEar audio data or AST pretrained model weights. Users must obtain these resources separately from their respective sources and comply with the applicable licensing terms.

See `LICENSE-NOTICE.md` for additional licensing information. An explicit project license should be selected before granting general reuse or redistribution rights.

## Modularization and Numerical Equivalence Validation

The modular HARPS implementation has been validated against the preserved reference implementation through complete experimental runs on the same dataset and computational environment.

### Structural and Unit Tests

The repository includes structural regression tests that compare the abstract syntax trees (ASTs) of the original top-level functions and classes against their modular counterparts.

Run the syntax checks and unit tests:

```bash
python -m compileall -q harps scripts tests run_experiments.py
python -m unittest discover -s tests -v
```

### Full Experimental Validation

Both the reference and modular implementations were executed using the same experimental configuration:

- **Dataset:** ShipsEar, 2,223 audio segments across 12 LOCO acoustic classes.
- **Random seeds:** 0, 1, 2, 3, 4.
- **Experimental protocol:** Reference-inclusive Leave-One-Class-Out (LOCO).
- **Hardware:** NVIDIA A100 80GB PCIe GPU.
- **Software environment:** Python 3.10.21, PyTorch 2.5.1+cu121, and Transformers 4.46.3.

### Numerical Equivalence Results

The following artifacts were compared between the reference and modular implementations:

| Artifact | Validation result |
|---|---|
| Model checkpoints (`.pt`) | Identical SHA256 hashes |
| Serialized models (`.joblib`) | Identical SHA256 hashes |
| NumPy arrays (`.npz`) | Identical SHA256 hashes |
| Evaluation metrics (`.csv`) | Numerically equivalent within `atol=rtol=1e-6` |
| Metadata and configuration (`.json`) | Equivalent after excluding environment-dependent paths and execution timings |
| Deep SVDD Wilcoxon signed-rank tests | Matching statistical results |

Both implementations generated 68,559 output files, including model checkpoints, serialized artifacts, predictions, evaluation metrics, and statistical summaries.

The comparison confirms numerical equivalence between the reference and modular implementations under the tested experimental configuration.

### Reproducing the Full Experiment

To execute the complete experiment on a compatible Linux/CUDA server, configure the following environment variables:

- `HARPS_DATA_ROOT`: Path to the experiment-ready ShipsEar dataset.
- `HARPS_AST_PATH`: Path to the pretrained AST model.
- `HARPS_OUTPUT_DIR`: Output directory for experimental results (optional).

Then execute:

```bash
bash scripts/run_full_experiments.sh
```

For an independent reproduction, use a fresh output directory to prevent previously cached features, model checkpoints, or completion markers from affecting the execution.

**Validation scope:** The equivalence assessment establishes consistency between the reference and modular implementations. It does not independently establish the statistical validity of the experimental design or the generalizability of the reported results.
