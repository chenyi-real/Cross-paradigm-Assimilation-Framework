# Case 2-Traffic: Physics-Informed Traffic Flow Prediction

This directory contains the Case 2 traffic-flow prediction experiments for the cross-paradigm assimilation framework. The implementation combines a Physics-Informed Neural Network (PINN) with a time-frequency branch and compares it with MLP, CNN, LSTM, and TCN baselines on connected-vehicle traffic data from the US101 scenario.

## Project Structure

```text
Case2-Traffic/
+-- configs/
|   +-- common.json                     # Shared training and physics settings
|   `-- us101_random0.05_*.json          # US101 experiment configurations
+-- data/
|   +-- pinn_data_us101_norm.csv.gz      # Preprocessed normalized US101 data
|   `-- pinn_scalers_us101.pk            # Normalization scalers
+-- FreTS.py                             # Frequency-domain branch
+-- main.py                              # Training and evaluation entry point
+-- model.py                             # PINN, fusion, and baseline models
+-- quality_metrics.py                   # MSE, RMSE, MAE, MAPE, PSNR, FSIM, etc.
+-- utils.py                             # Config, optimizer, and helper utilities
+-- requirements.txt                     # Python dependencies
+-- Result-*/                            # Reported Excel results
+-- loss_curve/                          # Loss-curve figures
+-- violin_map/                          # Violin-plot figures
`-- README.md
```

## Environment

Install dependencies in a Python environment with PyTorch:
```bash
cd Case2-Traffic
pip install -r requirements.txt
```

Main packages include `torch`, `pandas`, `scikit-learn`, `numpy`, `seaborn`, `matplotlib`, `opencv-python`, `scikit-image`, `phasepack-python`, and `pyFFTW`.

## Data

The training script expects preprocessed files under `data/`:

```text
data/
+-- pinn_data_us101_norm.csv.gz
`-- pinn_scalers_us101.pk
```

The CSV file should include normalized and raw columns used by `main.py`, including `time`, `distance`, `time_raw`, `distance_raw`, and the target mode such as `speed` and `speed_raw`.

## Configuration

Experiment settings are loaded from a JSON file passed through `--config`. Missing fields are filled from `configs/common.json`.

Common options include:

| Field | Description |
|:--|:--|
| `mode` | Prediction target, for example `["speed"]`. |
| `physical_model` | PDE residual function names, for example `["gs_speed_norm"]`. |
| `loss_pde` | Enables or disables the physics residual loss. |
| `train_sample_p` | Fraction of data sampled for training. |
| `n_epochs` | Number of training epochs. |
| `batch_size` | Batch size. |
| `lr`, `wd` | Learning rate and weight decay. |
| `fusion_hidden` | Hidden dimension of the PINN/time-frequency fusion MLP. |

## Running Experiments

Run commands from this directory:

```bash
cd Case2-Traffic
```

### Proposed PINN + Time-Frequency Model

```bash
python main.py --config configs/us101_random0.05_PINN_res.json --arch pinn_tf --pinn_tf_mode all
```

`--pinn_tf_mode` controls which branch is trained and evaluated:

| Mode | Description |
|:--|:--|
| `all` | Uses PINN, time-frequency branch, and fusion MLP. |
| `pinn` | Uses only the PINN branch. |
| `tf` | Uses only the time-frequency branch. |

### Baseline Models

```bash
python main.py --config configs/us101_random0.05_NN.json --arch mlp
python main.py --config configs/us101_random0.05_NN.json --arch cnn
python main.py --config configs/us101_random0.05_NN.json --arch lstm
python main.py --config configs/us101_random0.05_NN.json --arch tcn
```

### Useful Runtime Overrides

```bash
python main.py \
  --config configs/us101_random0.05_PINN_res.json \
  --arch pinn_tf \
  --pinn_tf_mode all \
  --n_epochs 3000 \
  --n_runs 20
```

## Outputs

Each run is written to a versioned directory under `logs/`:

```text
logs/<architecture>/<config_name>/version_<id>/
+-- checkpoint_*_best.pt
+-- metrics.json
+-- prediction.png
`-- predictions.csv.gz
```

Metrics include MSE, RMSE, MAE, MAPE, PSNR, and FSIM. The prediction heatmap is saved as `prediction.png`.

## Notes

- CUDA is used automatically when available.
- If `configs/common.json` is missing, `main.py` creates a default copy.
- The existing `Result-*`, `loss_curve/`, and `violin_map/` directories contain reported results and figures for baseline, ablation, and validity analysis experiments.
