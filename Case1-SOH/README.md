# Case 1-SOH: Battery SOH Estimation

This directory contains the Case 1-SOH experiments for battery state-of-health estimation. The code includes physics-informed SOH models, baseline comparison models, cross-domain fine-tuning, UDDS transfer/distillation workflows, and result-analysis utilities.

## Project Structure

```text
Case1-SOH/
+-- Model/
|   +-- Model.py                 # PINN, PINN variants, student model, and shared modules
|   +-- Compare_Models.py        # MLP, CNN, LSTM, TCN, and spike-based baselines
|   `-- test_custom_Model.py
+-- dataloader/
|   `-- dataloader.py            # Dataset reading, normalization, and DataLoader creation
+-- data/
|   +-- NASA/                    # NASA processed health-indicator files and preprocessing scripts
|   +-- MIT/                     # MIT health-indicator archive
|   +-- CALCE/                   # CALCE/CS2 preprocessing scripts
|   `-- UDDS/                    # UDDS health-indicator files
+-- data analysis/               # Sample-count and data-analysis helpers
+-- results_analysis/            # Result aggregation scripts
+-- utils/
|   `-- util.py                  # Logging, metrics, schedulers, and helper utilities
+-- main_NASA.py                 # NASA PINN experiment entry point
+-- main_MIT.py                  # MIT PINN experiment entry point
+-- main_CS2.py                  # CS2/CALCE PINN experiment entry point
+-- main_comparision.py          # Baseline comparison entry point
+-- main_adaptation - fine-tuning.py
|                                  # Cross-domain fine-tuning and UDDS distillation workflows
+-- predict_soh_distill.py        # Student-model prediction utility
+-- count parameters.py           # Parameter-count utility
`-- README.md
```

## Environment

This case uses PyTorch and common scientific Python packages:

```bash
conda create -n case1-soh python=3.10
conda activate case1-soh
pip install torch numpy pandas matplotlib scikit-learn tqdm openpyxl
```

Some comparison models import spike-neural-network packages. Install these only when running `Spikeformer`, `SpikeGRU`, or other spike-based baselines:

```bash
pip install snntorch spikingjelly
```

If your environment does not already provide `SeqSNN`, make sure that package or module is available on `PYTHONPATH` before running spike-based experiments.

## Data

The repository contains processed health-indicator examples under `data/`:

```text
data/
+-- NASA/
|   +-- HIs.xlsx
|   `-- HIs_temp.xlsx
+-- MIT/
|   `-- HIs.zip
+-- CALCE/
+-- UDDS/
|   +-- HIs_C1_W8.xlsx
|   +-- HIs_C2_W8.xlsx
|   `-- ...
```

Several training scripts currently reference external processed paths such as:

```text
hyx_data/NASA/new_out/
hyx_data/MIT/HIs/
hyx_data/CALCE/new_out/
hyx_data/UDDS/csv/
```

Before running experiments, either place the processed CSV/XLSX files at those paths or edit the corresponding `root` variables in the entry script. The dataloader expects health-indicator tables whose last column is the SOH/capacity target. If a `Cycle` column exists, it is removed before model input construction.

## Configuration

Most hyperparameters are command-line arguments defined inside each entry script.

Common options include:

| Field | Description |
|:--|:--|
| `batch_size` | Batch size for train/validation/test loaders. |
| `normalization_method` | Feature normalization method: `min-max` or `z-score`. |
| `epochs`, `early_stop` | Training length and early stopping patience. |
| `warmup_epochs`, `warmup_lr`, `lr`, `final_lr` | Learning-rate schedule settings. |
| `u_layers_num`, `u_hidden_dim` | Solution-network depth and hidden dimension for PINN models. |
| `F_layers_num`, `F_hidden_dim` | Dynamic-function network depth and hidden dimension. |
| `alpha`, `beta` | Weights for PDE and physics losses. |
| `save_folder`, `log_dir` | Output directory and log filename. |
| `pretrain_model` | Source-domain checkpoint path for fine-tuning or distillation. |
| `adaptation_lr`, `adaptation_epochs` | Fine-tuning/distillation learning rate and epochs. |

## Running Experiments

Run commands from this directory:

```bash
cd Case1-SOH
```

### NASA PINN Experiments

```bash
python main_NASA.py
```

The script iterates over `B0005`, `B0006`, `B0007`, and `B0018`, running 10 experiments for each battery.

### MIT PINN Experiments

```bash
python main_MIT.py
```

The script trains on MIT health indicators from `2017-05-12` and `2017-06-30`, using files whose numeric id is divisible by 5 as the test split.

### CS2/CALCE PINN Experiments

```bash
python main_CS2.py
```

The script iterates over `CS2_35`, `CS2_36`, `CS2_37`, and `CS2_38`.

### Baseline Comparison

```bash
python main_comparision.py
```

Baseline models are defined in `Model/Compare_Models.py`. The script includes MLP, CNN, LSTM, TCN, SpikeGRU, and related comparison workflows. Edit the calls in the `__main__` block to choose the model, dataset, and test battery.

### MIT-to-UDDS Fine-Tuning

The default `__main__` block of the fine-tuning script runs `FineTune_MIT2UDDS('Ours')`:

```bash
python "main_adaptation - fine-tuning.py"
```

Before running, check the `model_dir` assignment in `one_adaptation_task()` and make sure the pretrained source checkpoint exists.

### UDDS Distillation

In `main_adaptation - fine-tuning.py`, switch the `__main__` block from `FineTune_MIT2UDDS('Ours')` to:

```python
Distill_UDDS('Ours')
```

Then run:

```bash
python "main_adaptation - fine-tuning.py"
```

The distillation workflow saves the student checkpoint as `student_distill_model.pth`.

### Distilled Student Prediction

```bash
python predict_soh_distill.py
```

Update `model_path` and `dataset_root` in the script before prediction. The utility can locate UDDS files by cycle id, for example files containing `C12`, `C13`, or `C14`.

## Outputs

Experiment outputs are written under the `save_folder` configured by the active script. Common outputs include:

| Output | Description |
|:--|:--|
| `logging.txt` | Training, validation, test, and metric logs. |
| `model.pth` | Best baseline model checkpoint. |
| `finetune model.pth` | Fine-tuned model checkpoint. |
| `student_distill_model.pth` | Distilled student checkpoint. |
| `true_label.npy` | Ground-truth SOH values saved during testing. |
| `pred_label.npy` | Predicted SOH values saved during testing. |

Result-analysis scripts under `results_analysis/` aggregate NASA, MIT, CS2, and UDDS experiment outputs.

## Notes

- CUDA is used automatically when available, but some scripts set `CUDA_VISIBLE_DEVICES` explicitly. Adjust this value if your GPU layout differs.
- Several scripts contain hard-coded experiment loops and output paths. Review the `__main__` block before launching long runs.
- The filename `main_adaptation - fine-tuning.py` contains spaces, so quote it in shell commands.
- The code contains both current and legacy dataset paths; keep the selected entry script aligned with the processed data location you intend to use.
