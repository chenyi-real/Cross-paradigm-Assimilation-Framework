import torch
import os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATA_DIR = "./data/NASA_csv"
CYCLE_COL_NAME = "Discharge_Cycle"

CSV_FILES = ["B0005.csv", "B0006.csv", "B0007.csv", "B0018.csv"]
REQUIRED_COLS = ["Time(s)", "Voltage(V)", "Current(A)", "SOC"]

WINDOW = 64
STRIDE = 1 # 8
BATCH_SIZE = 128
NUM_WORKERS = 8

EPOCHS = 30
LR = 2e-3
PATIENCE = 10

RUNS = 10
SAVE_ROOT = 'runs/runs_baseline'
EXP_DIR = 'NASA-Ours'


EXP_DIR = os.path.join(SAVE_ROOT, EXP_DIR)
RESULTS_XLSX = os.path.join(EXP_DIR, "./NASA_results.xlsx")

PLOT = False
PLOTS_DIR =  os.path.join(EXP_DIR, "./plots")

MODELS_DIR = os.path.join(EXP_DIR, "./weights")

RESULTS_DIR = os.path.join(EXP_DIR, "./results")

