# 🐱‍🏍Cross-Paradigm Assimilation Framework

Title: **_Cross-paradigm assimilation framework unifies data-centric learning and knowledge-guided reasoning_**, which has been submitted.

This repository provides the code and supporting materials for a cross-paradigm assimilation framework that combines data-centric learning with knowledge-guided reasoning. The project is organized around battery prediction tasks and traffic-flow prediction tasks, with shared datasets, demo models, and plotting/analysis scripts.

The repository currently contains four case-study modules:

| Case | Task | Main Content |
|:--|:--|:--|
| `Case1-SOC/` | Battery state-of-charge estimation | SOC baselines, transfer learning, distillation, KalmanNet, SOC-PINN, and FreTS references. |
| `Case1-SOH/` | Battery state-of-health estimation | SOH PINN models, baselines, cross-domain fine-tuning, UDDS distillation, and result analysis. |
| `Case1-SOP/` | Battery state-of-power / battery prediction | NASA, CALCE, MIT, and UDDS workflows with two-branch modeling and distillation. |
| `Case2-Traffic/` | Traffic-flow prediction | Physics-informed traffic prediction with PINN, time-frequency branch, baselines, ablations, and reported results. |

Each case directory has its own README with task-specific setup and running instructions.

## Repository Structure

```text
Cross-paradigm-Assimilation-Framework/
+-- Case1-SOC/          # Battery SOC estimation experiments
+-- Case1-SOH/          # Battery SOH estimation experiments
+-- Case1-SOP/          # Battery SOP/battery prediction experiments
+-- Case2-Traffic/      # Traffic-flow prediction experiments
+-- data/               # Shared processed data examples
+-- demo/               # Lightweight teacher/student model demo code
+-- scripts/            # Plotting, result visualization, and data-cleaning utilities
`-- README.md
```

## Environment

The code is developed for Python 3.10 and primarily uses PyTorch. Different case studies have slightly different dependencies, so the safest setup is to create a Python 3.10 environment first and then install the root dependency file or the dependency file inside the case you want to run.

```bash
conda create -n cpaf python=3.10
conda activate cpaf
```

Install the common project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Case-specific dependency files are also kept for compatibility:

```bash
cd Case1-SOP
pip install -r requirements.txt
```

```bash
cd Case2-Traffic
pip install -r requirements.txt
```

Some optional baselines use additional spike-neural-network libraries such as `snntorch`, `spikingjelly`, and local or external `SeqSNN` modules. Install these only when running the corresponding spike-based models.

## Data

The top-level `data/` directory contains shared processed examples used by the battery-related cases:

```text
data/
+-- soc_data/
|   +-- NASA_csv/        # NASA SOC CSV files
|   `-- UDDS_HIS/        # UDDS health-indicator CSV files
`-- soh_data/
    +-- NASA/            # NASA SOH health-indicator files and preprocessing scripts
    +-- MIT/             # MIT SOH health-indicator archive
    +-- CALCE/           # CALCE preprocessing scripts
    `-- UDDS/            # UDDS SOH health-indicator XLSX files
```

Several case scripts also expect data under paths local to their case directories or historical processed-data paths such as `data/...`. Before running a script, check the active README and the data-root variables in that case's configuration or entry script.

Traffic data for `Case2-Traffic/` is expected under:

```text
Case2-Traffic/data/
+-- pinn_data_us101_norm.csv.gz
`-- pinn_scalers_us101.pk
```

## Demo

The `demo/` directory contains compact model definitions for teacher/student style experiments:

| File | Description |
|:--|:--|
| `demo.py` | Shared demo components and student encoders for SOH, SOC, and SOP. |
| `teacher_demo.py` | Teacher-side model components, including SOP two-branch structures. |
| `student_demo.py` | Student-side model components used for distillation demonstrations. |

These files are useful for inspecting the model design without navigating the full training pipelines. They are not a replacement for the complete case-specific workflows.

## Scripts

The `scripts/` directory contains plotting utilities.

```text
scripts/
+-- Draw_Error_Line/
|   +-- draw_error.py          # Draw error/scatter-line style result visualizations
|   `-- HD_scatter_out/        # Generated or collected scatter/error figures
`-- Draw_Violin/
    +-- violin_refactoring.py
    +-- violin_refactoring-batch.py
```

`Draw_Error_Line` is used for error-line and scatter-style visualizations across SOC, SOH, SOP, transfer, and distillation results. `Draw_Violin` is used for violin-plot summaries from Excel result files.


## Case Study Entry Points

### Case1-SOC

See `Case1-SOC/README.md` for details.

Typical commands:

```bash
cd Case1-SOC
python train_script/main.py
python train_script/main_mit.py
python train_script/main_transfer_udds.py
python train_script/main_distill_udds.py
```

### Case1-SOH

See `Case1-SOH/README.md` for details.

Typical commands:

```bash
cd Case1-SOH
python main_NASA.py
python main_MIT.py
python main_CS2.py
python main_comparision.py
python "main_adaptation - fine-tuning.py"
```

### Case1-SOP

See `Case1-SOP/README.md` for details.

Typical command:

```bash
cd Case1-SOP
python main.py
```

Experiments are controlled through `config.py`, including dataset selection, architecture selection, UDDS transfer, and distillation settings.

### Case2-Traffic

See `Case2-Traffic/README.md` for details.

Typical commands:

```bash
cd Case2-Traffic
python main.py --config configs/us101_random0.05_PINN_res.json --arch pinn_tf --pinn_tf_mode all
python main.py --config configs/us101_random0.05_NN.json --arch mlp
python main.py --config configs/us101_random0.05_NN.json --arch cnn
python main.py --config configs/us101_random0.05_NN.json --arch lstm
python main.py --config configs/us101_random0.05_NN.json --arch tcn
```

## Outputs

Outputs are written inside each case directory according to the active script or configuration. Common output types include:

| Output | Description |
|:--|:--|
| `checkpoints/`, `weights/` | Model checkpoints. |
| `results/`, `test_results/`, `logs/` | Metrics, predictions, and training logs. |
| `*.xlsx`, `metrics.json` | Aggregated numerical results. |
| `*.png` | Plots, heatmaps, violin plots, and error visualizations. |

The plotting scripts under `scripts/` consume these result files to generate publication-style summaries.

## Notes

- Run commands from the directory shown in each example unless the case README says otherwise.
- Many experiments use repeated runs and can take a long time on CPU. CUDA is used automatically in most scripts when available.
- Some scripts contain hard-coded data roots or checkpoint paths inherited from experiment runs. Check and adjust paths before launching a long experiment.
- The top-level README is a project map. For reproducible case-level details, use the README inside the corresponding case directory.
