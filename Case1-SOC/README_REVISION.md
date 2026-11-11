# SOC revision: implementation and evaluation scope

The original `model.py`, `FreTS.py`, data loaders and configuration files remain
the model backbone. `TwoBranchFramework` accepts windows of shape
`(batch, 64, 3)` and produces a scalar logit; apply sigmoid once for SOC fractions.
The constructor requires an output directory (`exp_dir`), not a window length.

## Running the existing experiments

Run from `Case1-SOC`, after configuring the files in `configs/`:

```bash
python run_experiment.py nasa
python run_experiment.py cs2
python run_experiment.py mit
python run_experiment.py transfer
python run_experiment.py distill
```

The launcher only supplies import paths; it does not change training data,
losses, splits, model dimensions or selection rules.

## Learned scalar gain

`kt_gain_model.py`, `kt_target_data.py`, `train_kalmannet_target.py` and
`predict_soc_kalmannet.py` implement a causal **SOC-observation** gain correction.
The frozen SOC predictor is the observation; current integration forms the prior.
This interface is not a voltage-observation EKF/UKF. Labels are used for fitting
and scoring, not as inference features. The historical two-output network is a
warm-start comparator, not the definition of this scalar gain model.

The gain fits the first 70% of original observer-validation trajectories per
unit; the remainder select warm/cold initialization and a trained epoch. Original
test trajectories do not select the gain. Gain feature scaling is train-only.
The frozen observer retains its legacy per-split input normalization; this is
not a claim of strictly online normalization. Available-capacity inputs require
an independently justified deployment protocol. Preserve these limitations
when reporting results.

Required inputs include existing observer checkpoints and the historical gain
checkpoint. Weights are not distributed by this code-only update. Set paths
using the following environment variables when their defaults do not apply:

| Variable | Default within this case | Meaning |
| --- | --- | --- |
| `CAF_DATASETS_ROOT` | `data` | Contains `CS2_csv` and `MIT/Processed_SOC` |
| `CAF_SOC_RUNS_ROOT` | `runs` | Contains `runs_baseline` and `runs_transfer` |
| `CAF_KALMAN_GAIN_CHECKPOINT` | `KalmanNet/checkpoints/final.pth` | Historical gain for warm initialization |

NASA CSVs are expected in `data/NASA_csv`; UDDS split caches in `data/UDDS_SOC`.

```bash
python train_kalmannet_target.py --family NASA --device cpu
python train_kalmannet_target.py --family NASA --evaluate-only --device cpu
python predict_soc_kalmannet.py --checkpoint selected.pt --input observations.csv --output corrected.csv
python verify_kalmannet_results.py --output results_kt/trained
```

The reference-free CSV interface expects one continuous trajectory with columns
`time_s`, `soc_observation`, `physical_increment`, and `current_A`.
`physical_increment` is integrated over the intervening raw current samples and
divided by `3600 * capacity_Ah`. Reset filtering at actual trajectory boundaries.
Saved artifacts include source-checkpoint hashes, trajectory identities, split
protocols and full-test metrics. A fixed gain of one reproduces the frozen SOC
observation. The verifier expects a completed all-family run, its original
checkpoints, plots and trace files; it is not a standalone test on the tables.

Figure-only favorable-segment selection and artificial voltage generation are
not part of this release candidate. The full-test scoring path remains intact.
The original `KalmanNet/features.py` reads reference SOC and the original
`KalmanNet/kalman_filter.py` uses an SOC-like innovation: those legacy files
must not be interpreted as a leakage-free voltage-observation benchmark.

The existing base trainers use MSE; the physics-informed reference is separately
under `SOC-PINN`. A unified implementation of every coordination replacement,
new baseline and loss-weight experiment in the revised tables has not been
identified in the reviewed source. Confirm those experiment-specific training
scripts before claiming full table reproduction.
