import torch
import os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATA_DIR = "./data/UDDS_SOC_v2"
CYCLE_COL_NAME = "Discharge_Cycle"   # UDDS的大cycle就是文件夹名，小cycle是文件名的序号

# DATASET_PARTS = ["A", "B", "C", "D"]
DATASET_PARTS = ["D", "C", "B", "A"]


CSV_FILES = [f"Cycling_{i}" for i in range(1, 5)] + [f"Cycling_{i}" for i in range(8, 15)]

print(CSV_FILES)
REQUIRED_COLS = ["Step_Time(s)", "Voltage(V)", "Current(A)", "SOC"]

WINDOW = 64 
STRIDE = 16 # 8
BATCH_SIZE = 512 # 128
NUM_WORKERS = 4

EPOCHS = 50 
LR = 1e-2
PATIENCE = 10

RUNS = 10 

METHOD_ID = 0 # 蒸馏只需要训练Ours就行了
METHOD = ['Ours', 'LSTM', 'CNN', 'MLP', 'TCN'][METHOD_ID]

SAVE_ROOT = 'runs/runs_distill_v3'

EXP_DIR = f'Distill-UDDS-{METHOD}'


EXP_DIR = os.path.join(SAVE_ROOT, EXP_DIR)
RESULTS_XLSX = os.path.join(EXP_DIR, "./Distill-UDDS_results.xlsx")

PLOT = False
PLOTS_DIR =  os.path.join(EXP_DIR, "./plots")

MODELS_DIR = os.path.join(EXP_DIR, "./weights")

PRETRAIN_WEIGHTS = f"runs/runs_transfer/MIT-Trans-UDDS-{METHOD}/weights/run_5/"

DISTILL_WEIGHT = 1e-3

RESULTS_DIR = './test_results/distill/UDDS/'