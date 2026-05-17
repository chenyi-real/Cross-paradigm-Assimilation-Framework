# Case 1-SOC: Battery SOC Estimation

This directory contains the Case 1-SOC experiments for battery state-of-charge estimation. It includes the main cross-paradigm assimilation implementation, transfer and distillation scripts for UDDS working-condition data, and several reference submodules such as KalmanNet, SOC-PINN, and FreTS.

## Project Structure

```text
Case1-SOC/
+-- configs/                    # Experiment configuration files
|   +-- config.py               # NASA baseline configuration
|   +-- config_cs2.py           # CS2 configuration
|   +-- config_mit.py           # MIT configuration
|   +-- config_transfer_udds.py # MIT-to-UDDS transfer configuration
|   `-- config_distill_udds.py  # UDDS distillation configuration
+-- train_script/               # Training and prediction entry points
|   +-- main.py                 # NASA repeated-run baseline entry point
|   +-- main_mit.py             # MIT training entry point
|   +-- main_cs2.py             # CS2 training entry point
|   +-- main_transfer_udds.py   # Transfer learning entry point
|   `-- main_distill_udds.py    # Distillation entry point
+-- data/                       # Expected dataset root
+-- dataloader.py               # SOC data loading utilities
+-- distill.py                  # Knowledge distillation logic
+-- FreTS.py                    # Frequency-domain branch
+-- model.py                    # Model definitions
+-- predict_soh_distill_v2.py   # Prediction/evaluation utility
+-- soh_dataloader.py           # SOH-related data loading utilities
+-- utils.py                    # Shared helpers
+-- KalmanNet/                  # KalmanNet reference implementation
+-- SOC-PINN/                   # SOC-PINN reference implementation
+-- FreTS/                      # FreTS reference implementation
`-- README.md
```

## Environment

This case uses PyTorch and common scientific Python packages. A typical environment can be created with:

```bash
conda create -n case1-soc python=3.10
conda activate case1-soc
pip install torch numpy pandas matplotlib scikit-learn tqdm openpyxl
```

The `FreTS/` submodule has its own dependency file:

```bash
cd FreTS
pip install -r requirements.txt
```

## Data

Configuration files define the expected data roots. Typical paths are:

```text
Case1-SOC/data/NASA_csv/
Case1-SOC/data/UDDS_SOC/
Case1-SOC/data/UDDS_SOC_v2/
```

NASA SOC experiments expect CSV files such as:

```text
B0005.csv
B0006.csv
B0007.csv
B0018.csv
```

UDDS experiments use cycle folders such as `Cycling_1` to `Cycling_14`, depending on the selected configuration. Required columns are defined in the active config, for example `Step_Time(s)`, `Voltage(V)`, `Current(A)`, and `SOC`.

## Configuration

Select the target experiment by editing the corresponding file under `configs/`.

Important settings include:

| Field | Description |
|:--|:--|
| `DATA_DIR` | Dataset root directory. |
| `CSV_FILES` | Files or cycle folders used by the experiment. |
| `WINDOW`, `STRIDE` | Sliding-window length and stride. |
| `BATCH_SIZE`, `EPOCHS`, `LR`, `PATIENCE` | Training hyperparameters. |
| `RUNS` | Number of repeated runs. |
| `METHOD` | Model family used in transfer or distillation experiments. |
| `PRETRAIN_WEIGHTS` | Path to pretrained weights for transfer/distillation. |
| `RESULTS_XLSX` | Aggregated result file. |

## Running Experiments

Run commands from this directory:

```bash
cd Case1-SOC
```

### NASA Baseline

```bash
python train_script/main.py
```

This entry point reads `configs/config.py`, trains repeated runs on the NASA CSV files, and writes aggregate and per-file metrics.

### MIT Training

```bash
python train_script/main_mit.py
```

Use `configs/config_mit.py` to adjust MIT dataset paths and training hyperparameters.

### CS2 Training

```bash
python train_script/main_cs2.py
```

Use `configs/config_cs2.py` to adjust CS2 dataset paths and training hyperparameters.

### MIT-to-UDDS Transfer

```bash
python train_script/main_transfer_udds.py
```

This workflow uses `configs/config_transfer_udds.py`. Set `PRETRAIN_WEIGHTS` to the pretrained MIT checkpoint before running.

### UDDS Distillation

```bash
python train_script/main_distill_udds.py
```

This workflow uses `configs/config_distill_udds.py`. Set `PRETRAIN_WEIGHTS` to the teacher or transferred checkpoint directory before running.

## Outputs

Results and checkpoints are written under the configured `SAVE_ROOT` and `EXP_DIR`, for example:

```text
runs/
+-- runs_baseline/
+-- runs_transfer/
`-- runs_distill_v3/
```

Common outputs include:

| Output | Description |
|:--|:--|
| `config.yaml` | Snapshot of the active configuration. |
| `weights/` | Saved model checkpoints by run. |
| `plots/` | Optional prediction plots when plotting is enabled. |
| `*_results.xlsx` | Aggregated RMSE and MAE metrics. |
| `test_results/` | Prediction or evaluation results for transfer/distillation workflows. |

## Reference Submodules

| Submodule | Description |
|:--|:--|
| `KalmanNet/` | Kalman filtering and neural Kalman estimation experiments. |
| `SOC-PINN/` | Physics-informed SOC estimation scripts and evaluation utilities. |
| `FreTS/` | Frequency-domain MLP time-series forecasting reference implementation. |

For FreTS-specific usage, see `FreTS/README.md`.

## Notes

- CUDA is used automatically when available.
- Some scripts import configuration modules directly; keep the intended config file and entry point paired together.
- Make sure dataset folders and pretrained checkpoint paths match the relative paths defined in the active configuration before starting a run.
