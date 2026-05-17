import torch
import os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATA_DIR = "./data/MIT"
CYCLE_COL_NAME = "cycle"

CSV_FILES = ["2017-05-12", "2017-06-30", "2018-04-12"]
REQUIRED_COLS = ["t", 'I', 'V', 'SOC']

WINDOW = 64
STRIDE = 1 # 16
BATCH_SIZE = 256
NUM_WORKERS = 8

EPOCHS = 50
LR = 2e-3
PATIENCE = 10

RUNS = 10
SAVE_ROOT = 'runs/runs_baseline'
EXP_DIR = 'MIT-Ours'


EXP_DIR = os.path.join(SAVE_ROOT, EXP_DIR)
RESULTS_XLSX = os.path.join(EXP_DIR, "./MIT_results.xlsx")

PLOT = False

PLOTS_DIR =  os.path.join(EXP_DIR, "./plots")

MODELS_DIR = os.path.join(EXP_DIR, "./weights")

