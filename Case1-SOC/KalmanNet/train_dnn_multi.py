"""
在原版 train_dnn_v2.py 基础上做了以下增强：
1) 数据集自动成对：基于 DATASET_CONFIG，自动把 *-50SOC 和 *-80SOC 组成一个数据集组（如 DST-0C 由 DST-0C-50SOC 与 DST-0C-80SOC 组成）。
2) 多文件不跨界滑窗：同一组内两个 CSV 分别滑窗，避免窗口跨文件边界；同时用“全局参数”做标准化，保证尺度统一。
3) 支持按数据集循环 n 次实验；每次保存：模型、配置、测试图，并把每次 RMSE/MAE 记录进一个 xlsx。
4) 目录结构：<数据集组名>/exp<编号>/run<编号>/... 例如：DST-0C/exp1/run1/。

你可以在 __main__ 部分修改 groups_to_run 和 n_runs 以批量运行。
"""

import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import DEVICE
from model import CNNGRUAttention


# -------------------- 超参 & 配置 --------------------
WINDOW_SIZE = 60
DATASET_CONFIG = {
    'DST-0C-50SOC': 2200,
    'DST-0C-80SOC': 900,
    'DST-25C-50SOC': 3000,
    'DST-25C-80SOC': 2000,
    'DST-45C-50SOC': 2000,
    'DST-45C-80SOC': 2200,

    'FUDS-0C-50SOC': 2000,
    'FUDS-0C-80SOC': 2000,
    'FUDS-25C-50SOC': 2500,
    'FUDS-25C-80SOC': 2200,
    'FUDS-45C-50SOC': 2100,
    'FUDS-45C-80SOC': 2000,

    'US06-0C-50SOC': 2200,
    'US06-0C-80SOC': 2000,
    'US06-25C-50SOC': 1500,
    'US06-25C-80SOC': 1210,
    'US06-45C-50SOC': 2100,
    'US06-45C-80SOC': 2000,
}

CSV_ROOT = './dataset/csv'
# OUTPUT_ROOT = './work_dir_multi_soc'  # 结果目录的根；最终会是 <OUTPUT_ROOT>/<group>/expX/runY/
OUTPUT_ROOT = './work_di_test'

# -------------------- 数据集定义（支持多文件不跨界滑窗） --------------------
class CALCESOCDatasetMulti(Dataset):
    def __init__(self, csv_paths, window_size=WINDOW_SIZE, use_zscore=True):
        super().__init__()
        self.window_size = window_size
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]

        # 先逐文件读原始数据，并按文件名应用各自 start_row
        raw_vs, raw_is, raw_ts, raw_ys = [], [], [], []
        per_file_arrays = []
        for path in csv_paths:
            df = pd.read_csv(path)
            name = os.path.splitext(os.path.basename(path))[0]
            start_row = DATASET_CONFIG.get(name, 1)
            df = df.iloc[start_row:].reset_index(drop=True)

            v = df["Voltage(V)"].values.astype(np.float32)
            i = df["Current(A)"].values.astype(np.float32)
            t = df["Test_Time(s)"].values.astype(np.float32)
            y = df["SOC(%)"].values.astype(np.float32)

            raw_vs.append(v); raw_is.append(i); raw_ts.append(t); raw_ys.append(y)
            per_file_arrays.append((v, i, t, y))

        # 计算全局标准化参数
        V = np.concatenate(raw_vs); I = np.concatenate(raw_is); T = np.concatenate(raw_ts); Y = np.concatenate(raw_ys)
        if use_zscore:
            v_mu, v_std = V.mean(), V.std() + 1e-8
            i_mu, i_std = I.mean(), I.std() + 1e-8
            t_mu, t_std = T.mean(), T.std() + 1e-8
            y_min, y_max = Y.min(), Y.max()
        else:
            v_min, v_max = V.min(), V.max()
            i_min, i_max = I.min(), I.max()
            t_min, t_max = T.min(), T.max()
            y_min, y_max = Y.min(), Y.max()

        # 按文件分别滑窗（不跨界）并拼接样本
        seqs, labels = [], []
        for (v, i, t, y) in per_file_arrays:
            if use_zscore:
                v = (v - v_mu) / v_std
                i = (i - i_mu) / i_std
                t = (t - t_mu) / t_std
                y = (y - y_min) / (y_max - y_min + 1e-8)
            else:
                v = np.clip((v - v_min) / (v_max - v_min + 1e-8), 0, 1)
                i = np.clip((i - i_min) / (i_max - i_min + 1e-8), 0, 1)
                t = np.clip((t - t_min) / (t_max - t_min + 1e-8), 0, 1)
                y = np.clip((y - y_min) / (y_max - y_min + 1e-8), 0, 1)

            Tlen = len(y)
            for idx in range(self.window_size - 1, Tlen):
                win_v = v[idx - self.window_size + 1: idx + 1]
                win_i = i[idx - self.window_size + 1: idx + 1]
                win_t = t[idx - self.window_size + 1: idx + 1]
                feat = np.stack([win_v, win_i, win_t], axis=1)
                seqs.append(feat)
                labels.append(y[idx])

        self.seqs = np.stack(seqs).astype(np.float32)
        self.labels = np.array(labels).astype(np.float32)
        print(f"[DATA] 多文件总样本: {self.seqs.shape[0]} 个, 窗长={self.window_size}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


# -------------------- 划分：最后 15% 作为测试，其余 85% -> 训练/验证 --------------------
def split_indices(idxs, seed=42):
    idxs = np.array(idxs)
    N = len(idxs)
    source_idxs = idxs
    perm = np.random.permutation(idxs)

    split_85 = int(N * 0.85)
    idxs_85 = perm[:split_85]
    idxs_15 = source_idxs[split_85:]  # 测试集（按顺序的最后 15%）

    split_train = int(len(idxs_85) * 0.8)
    train_idxs = idxs_85[:split_train]
    val_idxs = idxs_85[split_train:]
    test_idxs = idxs_15
    return train_idxs, val_idxs, test_idxs


# -------------------- 训练 + 测试（单次 run） --------------------
def train_one_run(csv_paths, work_dir,
                  batch_size=256, lr=1e-3, epochs=50, window_size=WINDOW_SIZE, patience=10):
    os.makedirs(work_dir, exist_ok=True)

    dataset = CALCESOCDatasetMulti(csv_paths, window_size, use_zscore=True)
    N = len(dataset)
    idxs = np.arange(N)
    train_idx, val_idx, test_idx = split_indices(idxs)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    device = torch.device(DEVICE)
    model = CNNGRUAttention(in_feats=3, cnn_channels=32, kernel_size=3,
                            pool_size=1, gru_hidden=64, out_feats=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.HuberLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    best_loss, patience_counter = float('inf'), 0
    best_model_path = os.path.join(work_dir, 'model_best.pth')

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).squeeze(-1)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 用最佳模型做测试
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    mae, rmse = test_and_save(model, test_loader, device, work_dir)

    # 保存本次 run 的配置
    config = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'csv_paths': csv_paths,
        'window_size': window_size,
        'batch_size': batch_size,
        'lr': lr,
        'epochs': epochs,
        'patience': patience,
        'model': 'CNNGRUAttention(in_feats=3, cnn_channels=32, kernel_size=3, pool_size=1, gru_hidden=64, out_feats=1)'
    }
    with open(os.path.join(work_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return mae, rmse, best_model_path


# -------------------- 测试并保存图 --------------------
def test_and_save(model, test_loader, device, work_dir):
    model.eval()
    y_true, y_pred, indices = [], [], []
    with torch.no_grad():
        for b_idx, (x, y) in enumerate(test_loader):
            idx = test_loader.dataset.indices[b_idx * test_loader.batch_size: b_idx * test_loader.batch_size + len(y)]
            indices.extend(idx)
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(-1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    indices = np.array(indices)
    y_true = np.array(y_true)[np.argsort(indices)]
    y_pred = np.array(y_pred)[np.argsort(indices)]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"测试集: MAE={mae:.6f}, RMSE={rmse:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Real")
    plt.plot(y_pred, label="Pred", linestyle='--')
    plt.xlabel('time')
    plt.ylabel('SOC')
    plt.title('Real vs Pred')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(work_dir, 'prediction_vs_true.png')
    plt.savefig(fig_path)
    plt.close()
    return mae, rmse


# -------------------- 数据集分组工具 --------------------
def build_groups_from_config():
    """把 *-50SOC 和 *-80SOC 自动成对，返回 {group_name: [file50, file80]}"""
    names = list(DATASET_CONFIG.keys())
    groups = {}
    for name in names:
        if name.endswith('-50SOC'):
            base = name[:-6]  # 去掉 '-50SOC'
            pair80 = f"{base}-80SOC"
            if pair80 in DATASET_CONFIG:
                groups[base] = [name, pair80]
    return groups  # e.g. {'DST-0C': ['DST-0C-50SOC','DST-0C-80SOC'], ...}


def csv_paths_for_group(group_name):
    files = [f"{group_name}-50SOC.csv", f"{group_name}-80SOC.csv"]
    return [os.path.join(CSV_ROOT, fn) for fn in files]


# -------------------- 批量运行：同一数据集组循环 n 次，并汇总到 xlsx --------------------
def run_experiments_for_group(group_name, n_runs=3,
                              batch_size=256, lr=1e-3, epochs=50, window_size=WINDOW_SIZE, patience=10):
    base_dir = os.path.join(OUTPUT_ROOT, group_name)
    os.makedirs(base_dir, exist_ok=True)
    

    results = []  # 用于写 xlsx: 列 [run, RMSE, MAE]
    csv_paths = csv_paths_for_group(group_name)

    for r in range(1, n_runs + 1):
        run_dir = os.path.join(base_dir, f"run{r}")
        mae, rmse, best_model_path = train_one_run(
            csv_paths=csv_paths,
            work_dir=run_dir,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            window_size=window_size,
            patience=patience,
        )
        results.append({'run': r, 'RMSE': rmse, 'MAE': mae})

    # 保存 xlsx
    df = pd.DataFrame(results, columns=['run', 'RMSE', 'MAE'])
   
    xlsx_path = os.path.join(OUTPUT_ROOT, f'{group_name}_results.xlsx')
    df.to_excel(xlsx_path, index=False)
    print(f"已保存结果到: {xlsx_path}")


# -------------------- 主入口 --------------------
if __name__ == "__main__":
    # 自动从 DATASET_CONFIG 生成所有成对的数据集组
    groups = build_groups_from_config()  # {'DST-0C': [...], 'DST-25C': [...], ...}

    # 你可以在这里挑选要跑的组；例如只跑 DST-0C：
    # groups_to_run = ['DST-0C']  # 改为 list(groups.keys()) 可一次跑全部
    groups_to_run = list(groups.keys())
    # 实验参数
    n_runs = 10      # 每个数据集循环次数
    batch_size = 256
    lr = 1e-3
    epochs = 50
    window_size = WINDOW_SIZE
    patience = 10
  
    for g in groups_to_run:
        print(f"\n========== Running group: {g} (n_runs={n_runs}) ==========")
        run_experiments_for_group(
            group_name=g,
            n_runs=n_runs,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            window_size=window_size,
            patience=patience,
        )
