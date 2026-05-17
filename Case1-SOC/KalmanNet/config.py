import torch

# 数据与输出路径
DATA_PATH = 'battery_data.xlsx'
OUTPUT_DIR = './checkpoints'

# 模型维度与差分输入维度
STATE_DIM = 2
OBS_DIM = 1

# 差分特征数量：F1（1维）、F2（1维）、F4（1维）
INPUT_DIM = 3

# GRU 隐藏态维度 h_dim
HIDDEN_DIM = 10 * (STATE_DIM**2 + OBS_DIM**2)

BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4

# V2（截断 BPTT）阶段的片段长度
SLICE_LEN = 100

# 预热与微调的 Epoch 数量
EPOCHS_WARMUP = 3   # V2 Warm-up
EPOCHS_FINETUNE = 3  # V1 Fine-tune

RANDOM_SEED = 42

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 卡尔曼滤波已知噪声协方差
Q_CLASSICAL = torch.diag(torch.tensor([1e-5, 1e-5])).to(DEVICE)   # 过程噪声方差
R_CLASSICAL = 1e-3   # 观测噪声方差

DNN_CHECKPOINT = 'checkpoints/cnn_gru_attention_soc.pth'