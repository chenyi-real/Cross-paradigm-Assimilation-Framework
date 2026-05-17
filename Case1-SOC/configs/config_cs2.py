import torch
import os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATA_DIR = "./data/CS2"
CYCLE_COL_NAME = "cycle"

CSV_FILES = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
REQUIRED_COLS = ["time_s", "Voltage(V)", "Current(A)", "SOC(%)"]

WINDOW = 64
STRIDE = 1 # 8
BATCH_SIZE = 128
NUM_WORKERS = 8

EPOCHS = 50
LR = 2e-3
PATIENCE = 10

RUNS = 10
SAVE_ROOT = 'runs/runs_baseline'
EXP_DIR = 'CS2-Ours'


EXP_DIR = os.path.join(SAVE_ROOT, EXP_DIR)
RESULTS_XLSX = os.path.join(EXP_DIR, "./CS2_results.xlsx")

PLOT = False
PLOTS_DIR =  os.path.join(EXP_DIR, "./plots")

MODELS_DIR = os.path.join(EXP_DIR, "./weights")

