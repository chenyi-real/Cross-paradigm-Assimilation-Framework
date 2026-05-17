import torch
import os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATA_DIR = "./data/UDDS_SOC"
CYCLE_COL_NAME = "Discharge_Cycle"   # UDDS的大cycle就是文件夹名，小cycle是文件名的序号

DATASET_PARTS = ["A", "B", "C", "D"] 

CSV_FILES = [f"Cycling_{i}" for i in range(1, 5)] + [f"Cycling_{i}" for i in range(8, 15)]

print(CSV_FILES)
REQUIRED_COLS = ["Step_Time(s)", "Voltage(V)", "Current(A)", "SOC"]

WINDOW = 64 
STRIDE = 8
BATCH_SIZE = 128
NUM_WORKERS = 8

EPOCHS = 50 
LR = 2e-3
PATIENCE = 10

RUNS = 10 

METHOD = 'Ours'

SAVE_ROOT = 'runs/runs_transfer'

EXP_DIR = f'MIT-Trans-UDDS-{METHOD}'


EXP_DIR = os.path.join(SAVE_ROOT, EXP_DIR)
RESULTS_XLSX = os.path.join(EXP_DIR, "./MIT-Trans-UDDS_results.xlsx")

PLOT = False
PLOTS_DIR =  os.path.join(EXP_DIR, "./plots")

MODELS_DIR = os.path.join(EXP_DIR, "./weights")

PRETRAIN_WEIGHTS = f"pretrained_weights/MIT-{METHOD}/weights/run_1/2017-05-12/best_model.pth"


RESULTS_DIR = './test_results/transfer/UDDS/'