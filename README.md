# HARPS: Asymmetric Reference Learning for Noise-Aware Maritime Acoustic Auditing

Official research implementation of **HARPS: Asymmetric Reference Learning for Noise-Aware Maritime Acoustic Auditing**, accepted for presentation at the **2026 IEEE International Conference on Big Data (IEEE BigData 2026)**.

This repository provides the modular Python implementation, experiment configurations, baseline methods, evaluation procedures, and supporting utilities for reproducing the HARPS experiments.

The implementation has been validated against the original research code through complete experimental execution and numerical equivalence testing.

## 1. Overview

HARPS is a noise-aware, one-class anomaly detection framework for underwater maritime acoustic monitoring.

The framework combines pretrained Audio Spectrogram Transformer (AST) representations with Deep Support Vector Data Description (Deep SVDD) and investigates asymmetric reference learning under clean and noise-corrupted acoustic conditions.

The experiments evaluate anomaly detection performance across different acoustic categories, feature representations, training configurations, and signal-to-noise ratios (SNRs).

The repository supports:

- Leave-One-Class-Out (LOCO) maritime acoustic anomaly detection.
- Frozen AST and MFCC feature representations.
- Deep SVDD with different clean, noisy, and paired training configurations.
- Asymmetric reference learning and consistency-based training.
- Traditional and deep-learning anomaly detection baselines.
- Gaussian and pink-noise evaluation conditions.
- Sensitivity analysis, representation geometry analysis, and computational benchmarking.
- Statistical evaluation and automated experiment summaries.

## 2. Repository Structure

```text
HARPS-Maritime-Acoustic-Auditing/
├── run_experiments.py
├── harps/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── noise.py
│   ├── features.py
│   ├── metrics.py
│   ├── baselines.py
│   ├── deep_svdd.py
│   ├── experiments.py
│   ├── reporting.py
│   ├── benchmarking.py
│   └── cli.py
├── scripts/
│   ├── download_ast.py
│   ├── download_shipsear.py
│   ├── check_dataset.py
│   ├── check_environment.py
│   └── run_full_experiments.sh
├── reference/
│   └── original_run_experiments.py
├── tests/
├── environment/
│   ├── environment.json
│   ├── pip_freeze.txt
│   └── conda_explicit.txt
├── requirements.txt
├── LICENSE-NOTICE.md
└── README.md
```

The main experimental implementation is contained in the `harps/` package.

The `scripts/` directory provides utilities for downloading research resources, checking the environment, and executing experiments.

The `reference/` and `tests/` directories are primarily intended for development, regression testing, and maintenance. They are not required for normal experiment execution.

## 3. Environment and Installation

### 3.1 Experimental Environment

The validated experiments were executed using the following environment:

| Component | Specification |
|---|---|
| Operating system | Linux |
| Python | 3.10.21 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.46.3 |
| PyOD | 3.6.6 |
| GPU | NVIDIA A100 80GB PCIe |
| CUDA build | 12.1 |

The recorded environment is available in `environment/environment.json`.

### 3.2 Clone the Repository

```bash
git clone https://github.com/taijun1101/HARPS-Maritime-Acoustic-Auditing.git

cd HARPS-Maritime-Acoustic-Auditing
```

### 3.3 Create the Conda Environment

```bash
conda create -n ast_deep_svdd python=3.10.21 -y

conda activate ast_deep_svdd
```

### 3.4 Install PyTorch

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 \
  torchaudio==2.5.1+cu121 \
  torchvision==0.20.1+cu121
```

### 3.5 Install Additional Dependencies

```bash
python -m pip install -r requirements.txt
```

The dependency configuration is primarily intended for compatible Linux systems with NVIDIA CUDA support.

For more precise environment reproduction, consult:

- `environment/environment.json`
- `environment/pip_freeze.txt`
- `environment/conda_explicit.txt`

Package availability and compatibility may vary across operating systems and CUDA versions.

### 3.6 Verify the Environment

```bash
python scripts/check_environment.py
```

This utility compares the installed environment against the recorded experimental configuration.

## 4. Dataset and Pretrained Model

### 4.1 ShipsEar Dataset

The experiments use the ShipsEar underwater acoustic dataset.

The experiment-ready dataset contains:

| Property | Value |
|---|---|
| Dataset | ShipsEar |
| Audio segments | 2,223 |
| Acoustic classes | 12 |
| Segment duration | 5 seconds |
| Sampling rate | 16,000 Hz |
| Experimental protocol | Reference-inclusive LOCO |

The dataset can be downloaded using the provided utility:

```bash
python scripts/download_shipsear.py \
  --output_dir ./data/DS3500 \
  --extract
```

Source dataset:

[ShipsEar / DS3500 on Hugging Face](https://huggingface.co/datasets/peng7554/DS3500)

**Important:** The downloaded `ShipsEar.zip` is not automatically equivalent to the experiment-ready `shipsear_5s_16k` dataset.

The download utility does not perform audio segmentation, resampling, or experimental preprocessing.

The experiment runner expects the following directory structure:

```text
shipsear_5s_16k/
├── group_1/
│   ├── class_1/
│   │   ├── segment_001.wav
│   │   └── segment_002.wav
│   └── class_2/
│       └── ...
└── group_2/
    └── ...
```

The directory names above are illustrative. The experiment runner reads WAV files recursively and derives acoustic class labels from the second directory component relative to `data_root`.

Ensure that the prepared dataset uses the same audio segmentation, sampling rate, class assignments, and file organization as the original experiment.

A verified end-to-end preprocessing utility for transforming the downloaded archive into the exact experiment-ready dataset is not currently included in this repository.

Verify the prepared dataset structure using:

```bash
python scripts/check_dataset.py \
  --data_root /path/to/shipsear_5s_16k
```

### 4.2 Pretrained Audio Spectrogram Transformer

HARPS uses a pretrained Audio Spectrogram Transformer (AST).

Model:

[MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

Download the model locally:

```bash
python scripts/download_ast.py \
  --output_dir ./ast_weight
```

The downloaded model is used as the pretrained acoustic feature extractor.

The frozen AST representation is obtained by averaging the first two special-token embeddings.

For the AST-based Deep SVDD configuration, the representation is passed through a single bias-free linear projection:

```text
Underwater Audio
      |
      v
Audio Spectrogram Transformer
      |
      v
Frozen AST Representation (768-D)
      |
      v
Linear Projection (768 -> 128)
      |
      v
Deep SVDD
      |
      v
Anomaly Score
```

The repository also supports MFCC-based representations and additional experimental configurations.

## 5. Experimental Protocol

### 5.1 Leave-One-Class-Out Anomaly Detection

HARPS is evaluated using a **reference-inclusive Leave-One-Class-Out (LOCO)** protocol.

For each held-out acoustic class:

1. The selected class is designated as the anomaly class.
2. All remaining classes are used as the normal reference set.
3. The anomaly detection model is trained using normal reference samples.
4. The complete dataset is evaluated, including both reference-class and held-out-class samples.
5. The procedure is repeated across all 12 acoustic classes.

Let the dataset be

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}.
$$

For a held-out class $c$, define the normal reference set as

$$
\mathcal{D}_{\mathrm{ref}}^{(c)}
=
\{x_i \mid y_i \neq c\},
$$

and the anomaly set as

$$
\mathcal{D}_{\mathrm{anom}}^{(c)}
=
\{x_i \mid y_i = c\}.
$$

The evaluation dataset is

$$
\mathcal{D}_{\mathrm{eval}}^{(c)}
=
\mathcal{D}_{\mathrm{ref}}^{(c)}
\cup
\mathcal{D}_{\mathrm{anom}}^{(c)}.
$$

This protocol does **not** use an 80/20 train-test split.

Because normal reference samples are also included in evaluation, the reported results should not be interpreted as performance on a fully unseen normal holdout set.

### 5.2 Noise Conditions

The experiments investigate anomaly detection under clean and noise-corrupted acoustic conditions.

The evaluation configurations include:

- Clean audio.
- Gaussian noise.
- Pink noise.
- Multiple signal-to-noise ratios.

The default noisy evaluation SNRs are:

\[
\mathrm{SNR}\in\{20,10,5\}\ \mathrm{dB}.
\]

Noise-related experiments examine the effects of acoustic corruption on anomaly ranking, representation consistency, and model robustness.

### 5.3 Evaluation Metrics

The experimental implementation produces anomaly detection metrics and statistical summaries, including:

- Area Under the Receiver Operating Characteristic Curve (AUROC).
- Area Under the Precision-Recall Curve (AUPRC).
- Additional ranking and detection metrics recorded by the experiment runner.
- Aggregated performance across LOCO classes.
- Statistical comparisons between supported Deep SVDD training configurations.

Higher anomaly scores indicate stronger evidence of anomalous acoustic behavior.

## 6. Running Experiments

### 6.1 Full Experiment Reproduction

Execute the following command from the repository root:

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

This command enables the full set of experimental options supplied for the research implementation.

Default settings include:

| Parameter | Default |
|---|---|
| Random seeds | 0, 1, 2, 3, 4 |
| Evaluation SNRs | 20, 10, 5 dB |
| Training epochs | 30 |
| LOCO classes | 12 |
| Batch size | 16 (explicitly specified above) |

Unspecified parameters use the defaults defined by the experiment runner.

### 6.2 Experiment Options

| Argument | Description |
|---|---|
| `--data_root` | Path to the prepared ShipsEar dataset |
| `--ast_model_path` | Path to the pretrained AST model |
| `--output_dir` | Directory for experiment outputs |
| `--run_all` | Run the baseline experiment grid |
| `--run_asymmetric` | Enable asymmetric learning experiments |
| `--run_mfcc_svdd` | Enable MFCC-based Deep SVDD experiments |
| `--run_pink` | Enable pink-noise experiments |
| `--run_sensitivity` | Enable sensitivity analysis |
| `--run_eigen` | Enable eigenvalue-related analysis |
| `--benchmark_cost` | Enable computational cost benchmarking |
| `--run_geometry_comparison` | Enable representation geometry comparison |
| `--batch_size` | Set the training batch size |

The baseline experiment grid includes the methods supported by the supplied implementation, including kNN and Mahalanobis.

For the exact behavior of each option and additional configuration parameters, refer to `harps/cli.py` and `harps/config.py`.

### 6.3 Shell Script Execution

The repository also provides a shell script for executing the full experiment.

Configure the required environment variables:

```bash
export HARPS_DATA_ROOT="/path/to/shipsear_5s_16k"

export HARPS_AST_PATH="./ast_weight"

export HARPS_OUTPUT_DIR="./results"
```

Then run:

```bash
bash scripts/run_full_experiments.sh
```

The shell script is intended for compatible Linux/CUDA environments.

### 6.4 Reproduction Recommendations

For independent reproduction:

- Use the experiment-ready ShipsEar dataset with matching audio segments and labels.
- Use the same pretrained AST checkpoint and software dependencies.
- Preserve the experimental random seeds and evaluation conditions.
- Use a fresh output directory to avoid unintended reuse of cached results.
- Check the generated experiment configuration and dataset manifest before interpreting the results.

The runner may reuse existing feature caches, model artifacts, and `_SUCCESS.json` completion markers when they are present.

## 7. Experimental Outputs

All outputs are stored in the directory specified by `--output_dir`.

The main output structure includes:

```text
results/
├── dataset_manifest.csv
├── loco_sample_counts.csv
├── ast_metadata.json
├── experiment_config.json
├── baselines/
├── deep_svdd/
├── sensitivity/
├── feature_cache/
└── summary/
    ├── baselines/
    ├── deep_svdd/
    └── sensitivity/
```

### 7.1 Dataset and Configuration

| File | Description |
|---|---|
| `dataset_manifest.csv` | Dataset file information and class assignments |
| `loco_sample_counts.csv` | Sample counts for each LOCO task |
| `ast_metadata.json` | AST model metadata |
| `experiment_config.json` | Recorded experiment configuration |

### 7.2 Model and Evaluation Results

The `baselines/`, `deep_svdd/`, and `sensitivity/` directories contain experiment-specific artifacts, which may include:

- Evaluation metrics.
- Per-sample anomaly scores and predictions.
- Trained model checkpoints.
- Serialized baseline models.
- Feature representations and analysis outputs.
- Experiment completion metadata.

### 7.3 Statistical Summaries

The `summary/` directory contains aggregated experimental results.

Representative outputs include:

| File | Description |
|---|---|
| `overall_metrics_across_loco.csv` | Aggregated anomaly detection metrics across LOCO classes |
| `all_task_metrics.csv` | Collected metrics from individual experimental tasks |
| `paired_wilcoxon_tests.csv` | Paired Wilcoxon signed-rank comparisons for supported Deep SVDD configurations |

The Wilcoxon analysis compares selected Deep SVDD training configurations using paired LOCO-class performance, with Holm-adjusted p-values.

It does not automatically provide statistical comparisons between HARPS and every baseline method.

## 8. Validation and Testing

The modular HARPS implementation has been validated against the preserved original research implementation through complete experimental runs.

Model artifacts, numerical outputs, evaluation metrics, and supported statistical summaries were compared to establish implementation equivalence under the tested configuration.

These developer-side equivalence checks are not required for normal use of the repository.

Users may optionally perform syntax checks and run the included unit tests:

```bash
python -m compileall -q \
  harps scripts tests run_experiments.py

python -m unittest discover -s tests -v
```

The unit tests are intended to detect software regressions. They do not replace full experimental reproduction.

## 9. Citation

If you use HARPS or its implementation in your research, please cite:

**You-Zhen Lai and Cheng-Te Li**, "HARPS: Asymmetric Reference Learning for Noise-Aware Maritime Acoustic Auditing," *2026 IEEE International Conference on Big Data (IEEE BigData)*, 2026.

```bibtex
@inproceedings{lai2026harps,
  title     = {HARPS: Asymmetric Reference Learning for Noise-Aware Maritime Acoustic Auditing},
  author    = {Lai, You-Zhen and Li, Cheng-Te},
  booktitle = {2026 IEEE International Conference on Big Data (IEEE BigData)},
  year      = {2026}
}
```

The citation will be updated when the final publication metadata, including DOI and page numbers, becomes available.

## 10. License and Attribution

This repository does not redistribute ShipsEar audio data or pretrained AST model weights.

Users must obtain these resources separately from their respective sources and comply with the applicable licensing and usage terms.

- ShipsEar dataset: [peng7554/DS3500](https://huggingface.co/datasets/peng7554/DS3500)
- Pretrained AST model: [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

See `LICENSE-NOTICE.md` for additional information about attribution and licensing.

Unless an explicit software license is provided, no general permission to reuse, modify, or redistribute the repository's source code should be assumed.
