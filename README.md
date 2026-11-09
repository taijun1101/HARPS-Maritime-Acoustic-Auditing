# HARPS: Asymmetric Reference Learning for Noise-Aware Maritime Acoustic Auditing

Official research implementation of **HARPS: Asymmetric Reference Learning for Noise-Aware Maritime Acoustic Auditing**, accepted for presentation at the **2026 IEEE International Conference on Big Data (IEEE BigData 2026)**.

This repository provides the modular Python implementation, experiment configurations, baseline methods, evaluation procedures, and supporting utilities for reproducing the HARPS experiments.

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
├── environment/
│   ├── environment.json
│   ├── pip_freeze.txt
│   └── conda_explicit.txt
├── .gitignore
├── requirements.txt
├── LICENSE-NOTICE.md
└── README.md

```

The main experimental implementation is contained in the `harps/` package.

The `scripts/` directory provides utilities for downloading research resources, checking the environment, and executing experiments.

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

The dataset used in the HARPS experiments has the following characteristics:

| Property | Value |
|---|---|
| Dataset | ShipsEar |
| Audio segments | 2,223 |
| Acoustic classes | 12 |
| Segment duration | 5 seconds |
| Sampling rate | 16,000 Hz |
| Experimental protocol | Reference-inclusive LOCO |

Download and extract the dataset using:

```bash
python scripts/download_shipsear.py \
  --output_dir ./data/DS3500 \
  --extract
```

Dataset source: [ShipsEar / DS3500 on Hugging Face](https://huggingface.co/datasets/peng7554/DS3500)

After extraction, the experiment-ready dataset is located at:

```text
data/DS3500/ShipsEar/shipsear_5s_16k/
├── 0/
├── 1/
├── 2/
├── 3/
├── 4/
├── label_list.txt
├── train_list.txt
├── test_list.txt
└── ...
```
The ShipsEar dataset follows a two-level directory structure. The five top-level directories (`0`–`4`) contain subdirectories corresponding to acoustic categories.

Each acoustic class is identified by combining the two directory levels. For example, audio files under `0/0/` are assigned to class `0_0`.

The HARPS experiments use 12 distinct acoustic classes defined by these directory combinations.

The dataset can be used directly after extraction without additional audio segmentation or resampling.

Verify the dataset directory using:

```bash
python scripts/check_dataset.py \
  --data_root ./data/DS3500/ShipsEar/shipsear_5s_16k
```

Use this same directory as the `--data_root` argument when executing `run_experiments.py`.
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
\mathcal{D} = \lbrace (x_i, y_i) \rbrace_{i=1}^{N}.
$$

For a held-out class $c$, define the normal reference set as

$$
\mathcal{D}_{\mathrm{ref}}^{(c)} = \lbrace x_i \mid y_i \neq c \rbrace.
$$

and the anomaly set as

$$
\mathcal{D}_{\mathrm{anom}}^{(c)} = \lbrace x_i \mid y_i = c \rbrace.
$$

The evaluation dataset is

$$
\mathcal{D}_{\mathrm{eval}}^{(c)} = \mathcal{D}_{\mathrm{ref}}^{(c)} \cup \mathcal{D}_{\mathrm{anom}}^{(c)}.
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

$$
\mathrm{SNR} \in \lbrace 20, 10, 5 \rbrace\ \mathrm{dB}.
$$


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
  --data_root ./data/DS3500/ShipsEar/shipsear_5s_16k \
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
export HARPS_DATA_ROOT="./data/DS3500/ShipsEar/shipsear_5s_16k"

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

- Use the ShipsEar dataset downloaded and extracted from the specified Hugging Face repository.
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

## 8. Citation

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

## 9. License and Attribution

This repository does not redistribute ShipsEar audio data or pretrained AST model weights.

Users must obtain these resources separately from their respective sources and comply with the applicable licensing and usage terms.

- ShipsEar dataset: [peng7554/DS3500](https://huggingface.co/datasets/peng7554/DS3500)
- Pretrained AST model: [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

See `LICENSE-NOTICE.md` for additional information about attribution and licensing.

Unless an explicit software license is provided, no general permission to reuse, modify, or redistribute the repository's source code should be assumed.
