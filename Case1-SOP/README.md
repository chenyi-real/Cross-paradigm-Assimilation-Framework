# Case 1-SOP: Battery SOP/SOH Prediction

This directory contains the Case 1-SOP experiments for battery prediction under the cross-paradigm assimilation framework. The code supports single-dataset training on NASA, CALCE, and MIT battery datasets, and a whole-model transfer/distillation workflow on UDDS working-condition data.

## Project Structure

```text
Case1-SOP/
+-- config.py              # Dataset, model, training, transfer, and distillation settings
+-- dataloader.py          # Dataset loading and sliding-window construction
+-- FreTS.py               # Frequency-domain branch
+-- main.py                # Main experiment entry point
+-- model.py               # Baseline and two-branch model definitions
+-- train.py               # Training, validation, testing, and metric utilities
+-- requirements.txt       # Python dependencies
`-- README.md
```

Expected data directories are referenced in `config.py`:

```text
Case1-SOP/
+-- NASA/
+-- CALCE/
+-- MIT/
`-- UDDS_condition/
```

## Environment

The experiments were prepared with Python 3.10.18. Main dependencies are:

| Package | Version |
|:--|:--|
| torch | 2.9.0 |
| numpy | 2.4.0 |
| pandas | 2.3.3 |
| matplotlib | 3.10.8 |
| tqdm | 4.67.1 |
| scikit-learn | 1.7.2 |

Create and activate a conda environment:

```bash
conda create -n case1-sop python=3.10.18
conda activate case1-sop
pip install -r requirements.txt
```

## Configuration

All major experiment options are defined in `config.py`.

Important fields:

| Field | Description |
|:--|:--|
| `DATASET_NAME` | Selects `NASA`, `CALCE`, or `MIT` for single-dataset training. |
| `NETWORK` | Selects the baseline or proposed model: `mlp`, `cnn`, `lstm`, `tcn`, or `two_branch`. |
| `WHOLE_ENABLE` | Enables the UDDS whole-model workflow. When `False`, the selected single-dataset experiment is used. |
| `DISTILL_ENABLE` | Enables student-model distillation in the UDDS whole-model workflow. |
| `WHOLE_ARCH` | Architecture used in the whole-model workflow. Use `two_branch` for the proposed method. |
| `UDDS_GROUP` | Selects the UDDS train/test split group: `A`, `B`, `C`, or `D`. |
| `RUNS` | Number of repeated runs. |

## Running Experiments

Run all commands from this directory:

```bash
cd Case1-SOP
```

### Single-Dataset Training

Set `WHOLE_ENABLE = False` in `config.py`, then choose the dataset and model:

```python
DATASET_NAME = "MIT"      # or "NASA", "CALCE"
NETWORK = "two_branch"    # or "mlp", "cnn", "lstm", "tcn"
WHOLE_ENABLE = False
```

Start training:

```bash
python main.py
```

### UDDS Teacher Training

1. Pretrain on a source dataset, for example MIT:

```python
DATASET_NAME = "MIT"
NETWORK = "two_branch"
WHOLE_ENABLE = False
```

2. Run `python main.py` and copy the produced `best.pt` checkpoint path to `WHOLE_A_CKPT`.

3. Train the UDDS teacher model:

```python
WHOLE_ENABLE = True
DISTILL_ENABLE = False
WHOLE_ARCH = "two_branch"
UDDS_GROUP = "A"
```

4. Run:

```bash
python main.py
```

### UDDS Student Distillation

After the teacher checkpoint is available, enable distillation:

```python
WHOLE_ENABLE = True
DISTILL_ENABLE = True
WHOLE_ARCH = "two_branch"
UDDS_GROUP = "A"
```

Then run:

```bash
python main.py
```

By default, the code loads the teacher checkpoint from `./checkpoints_whole/two_branch/<UDDS_GROUP>/run_1/best.pt`.

## Outputs

Training artifacts are saved automatically:

| Output | Description |
|:--|:--|
| `checkpoints/` | Best and epoch checkpoints for NASA, CALCE, and MIT experiments. |
| `checkpoints_whole/` | UDDS teacher-model checkpoints. |
| `checkpoints_whole_distill/` | UDDS distillation checkpoints. |
| `results/` | Per-dataset and per-file metrics. |
| `results_whole/` | UDDS teacher-model metrics. |
| `results_whole_distill/` | UDDS student distillation metrics. |
| `*.xlsx` | Aggregated RMSE, MAE, and, for MIT, MAPE results. |

## Notes

- When `WHOLE_ENABLE = True`, set `WHOLE_ARCH = "two_branch"` unless you intentionally run another architecture.
- `main.py` resumes run numbering from existing result files where supported.
- GPU is used automatically when CUDA is available; otherwise the code runs on CPU.
