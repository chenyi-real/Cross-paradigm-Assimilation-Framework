import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_NAME = "CALCE"

# 选择要训练的网络：["mlp", "cnn", "lstm", "tcn", "two_branch"]
NETWORK = "two_branch"

# 打开训练双分路整网（默认直接使用 UDDS 数据）；关闭保持原有单路
WHOLE_ENABLE = True

DISTILL_ENABLE = False
DISTILL_WEIGHT = 1e-3

# 选择要训练的网络：["mlp", "cnn", "lstm", "tcn", "two_branch", "student"]
WHOLE_ARCH = "two_branch"

# - 验证集固定为 Cycling_1, Cycling_2
# - A: 测试集 = 97-98
# - B: 测试集 = 96-97
# - C: 测试集 = 91-92
# - D: 测试集 = 90-91
UDDS_GROUP = "A"

UDDS_DIR = "./UDDS_condition"

WHOLE_A_CKPT = f"./checkpoints/{WHOLE_ARCH}/best.pt"

DATA_ROOTS = {
    "NASA": "./NASA",
    "CALCE": "./CALCE",
    "MIT": "./MIT",
}
DATA_DIR = DATA_ROOTS[DATASET_NAME]

WINDOW_NASA = 16
WINDOW_MIT = 64
STRIDE_NASA = 8
STRIDE_MIT = 16
WINDOW_CALCE = 16
STRIDE_CALCE = 8
WINDOW_UDDS = 64
STRIDE_UDDS = 16
BATCH_SIZE = 512
NUM_WORKERS = 0

EPOCHS = 30
LR = 0.001
PATIENCE = 5

RUNS = 10
EXP_DIR = 'exp1'

CAPACITY_NASA = 2.0
CAPACITY_MIT = 1.1
CAPACITY_CALCE = 1.1
COULOMB_EFFICIENCY = 1
CAPACITY_CONDITION = 4.93
