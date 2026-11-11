# NWAD-2.0

An open-set network intrusion detection pipeline based on the UNSW-NB15 dataset. The main workflow prepares reproducible data splits, trains a known-class MLP, calibrates rejection thresholds using validation data, and evaluates the frozen protocol on the held-out test split.

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/oscrjz/NWAD-2.0.git
cd NWAD-2.0
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch is used for model training and inference. For a CUDA-enabled installation, follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose the build for your system.

## Dataset and preprocessing

Download the four UNSW-NB15 raw CSV files and place them here, using these exact names:

```text
datasets/raw/
  UNSW-NB15_1.csv
  UNSW-NB15_2.csv
  UNSW-NB15_3.csv
  UNSW-NB15_4.csv
```

Generate the processed splits:

```bash
python preprocessing_data.py
```

This creates `datasets/processed/config_A/` and `datasets/processed/config_C/`, each with `without_smote/` and `with_smote/` variants. Each variant contains `train.csv`, `validation.csv`, `test.csv`, and `metadata.json`. SMOTENC is applied to training data only.

The split protocol reserves DoS as a pseudo-unknown class: it is excluded from training and appears in validation and test. Other unknown attack categories are held for test. Do not use the test split to select the model or calibrate rejection thresholds.

## Feature sets and configurations

The training and evaluation scripts support four dataset choices:

- `A` and `A-Smote`: Configuration A, with or without SMOTENC training data.
- `C` and `C-Smote`: Configuration C, with or without SMOTENC training data.

The model's known classes are configuration-specific. Available feature sets are:

- `behavioral` (default): selected flow-behavior features.
- `shortcut`: protocol, service, port, and state features for shortcut-learning analysis.
- `combined`: both feature groups.

Use the same dataset configuration and feature set throughout an experiment. Training saves these settings with the checkpoint; downstream scripts load the experiment configuration.

## Workflow

### 1. Train and select a model

For example:

```bash
python train.py --dataset A --feature-set behavioral --loss ce --experiment-name A_behavioral_ce --no-show
```

Training fits preprocessing on the training split only, selects the best checkpoint by validation Macro-F1, and uses early stopping. To use the SMOTENC variant, choose `--dataset A-Smote` (or `C-Smote`). See `python train.py --help` for all options, including the center-loss objective and training settings.

With `--experiment-name`, artifacts are saved under `checkpoints/<experiment-name>/`:

- `best.pt` and `last.pt`: model and training checkpoints.
- `preprocessor.joblib` and `config.json`: fitted preprocessing and experiment configuration.
- `history.csv` and `confusion_matrix_val.png`: training history and best-model validation results.

### 2. Calibrate open-set rejection on validation

After model selection, fit class centroids and rejection thresholds using the validation split:

```bash
python calibrate.py --experiment checkpoints/A_behavioral_ce --metric euclidean --no-show
```

Supported distance metrics are `euclidean` and `mahalanobis`. Calibration compares confidence-based, distance-based, and combined rejection rules while constraining the false rejection rate of known validation samples. It saves the frozen thresholds and calibration figures in the experiment directory. To compare a different training run, use that run's experiment directory.

### 3. Evaluate

During development, evaluate on validation:

```bash
python test.py --experiment checkpoints/A_behavioral_ce --metric euclidean --split validation --no-show
```

Evaluate the held-out test split only once model selection and calibration are complete. The `--final` flag is required to make this deliberate:

```bash
python test.py --experiment checkpoints/A_behavioral_ce --metric euclidean --split test --final --no-show
```

`test.py` loads the dataset, feature set, checkpoint, and fitted preprocessor from the experiment and checks that the calibration matches the selected checkpoint. It reports closed-set results for known classes and open-set results for the calibrated rejection rules. Results and figures are written to the experiment directory. Use `--help` on each script for available options.

### 4. Visualize learned embeddings (optional)

Project training and validation embeddings with PCA:

```bash
python visualize_embeddings.py --experiment checkpoints/A_behavioral_ce --method pca --no-show
```

UMAP is also available with `--method umap` or `--method both`.

## Repository overview

- `preprocessing_data.py`: raw-data cleaning, splitting, and optional SMOTENC.
- `train.py`, `calibrate.py`, `test.py`: model training, validation calibration, and final evaluation.
- `data/`, `models/`, `utilis/`: data loading, model, loss, evaluation, and open-set utilities.
- `visualize_embeddings.py`: optional embedding visualizations.
- `src/` and `notebooks/`: feature analysis, exploratory work, and legacy utilities.
- `src/legacy/attackdetect.py`: packet-capture demo; requires Scapy and, on Windows, Npcap. It is not part of the model training/evaluation workflow.
