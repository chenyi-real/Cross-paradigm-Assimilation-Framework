"""
优化后的 DNN 神经网络训练脚本 (预测SOC)
改进点：
1. 使用 z-score 标准化，避免分布漂移。
2. 随机划分训练/验证/测试集，避免分布不一致。
3. 模型结构：CNN + BiGRU + Attention + Dropout + BatchNorm。
4. 加入 EarlyStopping 和 ReduceLROnPlateau。
5. 使用 HuberLoss 替代 MSE，更鲁棒。
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import DEVICE
from model import CNNGRUAttention


WINDOW_SIZE = 60
DATASET_CONFIG={'DST-0C-50SOC': 2200,
                'DST-0C-80SOC': 900,
                'DST-25C-50SOC': 3000,
                'DST-25C-80SOC': 2000,
                'DST-45C-50SOC': 2000,
                'DST-45C-80SOC': 2200,

                'FUDS-0C-50SOC': 2000,
                'FUDS-0C-80SOC': 2000,
                'FUDS-25C-50SOC':2500,
                'FUDS-25C-80SOC':2200,
                'FUDS-45C-50SOC':2100,
                'FUDS-45C-80SOC':2000,

                'US06-0C-50SOC': 2200,
                'US06-0C-80SOC': 2000,
                'US06-25C-50SOC':1500,
                'US06-25C-80SOC':1210,
                'US06-45C-50SOC':2100,
                'US06-45C-80SOC':2000,
                }


# -------------------- 数据集定义 --------------------
class CALCESOCDataset(Dataset):
    def __init__(self, csv_path, window_size=WINDOW_SIZE, start_row=1, use_zscore=True):
        super(CALCESOCDataset, self).__init__()
        self.window_size = window_size
        self.data = pd.read_csv(csv_path)
        self.data = self.data.iloc[start_row:].reset_index(drop=True)

        self.voltage = self.data["Voltage(V)"].values.astype(np.float32)
        self.current = self.data["Current(A)"].values.astype(np.float32)
        self.time = self.data["Test_Time(s)"].values.astype(np.float32)
        self.soc = self.data["SOC(%)"].values.astype(np.float32)

        # ---------- 归一化 ----------
        if use_zscore:
            self.voltage = (self.voltage - self.voltage.mean()) / (self.voltage.std() + 1e-8)
            self.current = (self.current - self.current.mean()) / (self.current.std() + 1e-8)
            self.time = (self.time - self.time.mean()) / (self.time.std() + 1e-8)
            self.soc = (self.soc - self.soc.min()) / (self.soc.max() - self.soc.min())
        else:
            self.voltage = np.clip((self.voltage - self.voltage.min()) / (self.voltage.max() - self.voltage.min()), 0, 1)
            self.current = np.clip((self.current - self.current.min()) / (self.current.max() - self.current.min()), 0, 1)
            self.time = np.clip((self.time - self.time.min()) / (self.time.max() - self.time.min()), 0, 1)
            self.soc = np.clip((self.soc - self.soc.min()) / (self.soc.max() - self.soc.min()), 0, 1)

        # ---------- 滑动窗口 ----------
        self.seqs, self.labels = [], []
        T = len(self.soc)
        for idx in range(self.window_size - 1, T):
            win_v = self.voltage[idx - self.window_size + 1:idx + 1]
            win_i = self.current[idx - self.window_size + 1:idx + 1]
            win_t = self.time[idx - self.window_size + 1:idx + 1]
            feat = np.stack([win_v, win_i, win_t], axis=1)  # (W,3)
            self.seqs.append(feat)
            self.labels.append(self.soc[idx])

        self.seqs = np.stack(self.seqs).astype(np.float32)
        self.labels = np.array(self.labels).astype(np.float32)
        print(f"[DATA] 样本: {self.seqs.shape[0]} 个, 窗长={self.window_size}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]

class CALCESOCDatasetMulti(Dataset):
    def __init__(self, csv_paths, window_size=WINDOW_SIZE, use_zscore=True):
        super(CALCESOCDatasetMulti, self).__init__()
        self.window_size = window_size

        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]

        # -------- 先逐文件读原始数据，并按文件名应用各自 start_row --------
        raw_vs, raw_is, raw_ts, raw_ys = [], [], [], []
        per_file_arrays = []  # 保存每个文件的四元组(np.array)，用于二次滑窗

        for path in csv_paths:
            df = pd.read_csv(path)

            # 根据文件名决定 start_row（如命中 DATASET_CONFIG）
            name = os.path.splitext(os.path.basename(path))[0]
            start_row = DATASET_CONFIG.get(name, 1)
            df = df.iloc[start_row:].reset_index(drop=True)

            v = df["Voltage(V)"].values.astype(np.float32)
            i = df["Current(A)"].values.astype(np.float32)
            t = df["Test_Time(s)"].values.astype(np.float32)
            y = df["SOC(%)"].values.astype(np.float32)

            raw_vs.append(v); raw_is.append(i); raw_ts.append(t); raw_ys.append(y)
            per_file_arrays.append((v, i, t, y))

        # -------- 计算全局归一化/标准化参数 --------
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

        # -------- 按文件分别滑窗（不跨界）并拼接样本 --------
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
            # 仅在本文件范围内滑窗；不会使用到“跨文件的前一段”
            for idx in range(self.window_size - 1, Tlen):
                win_v = v[idx - self.window_size + 1: idx + 1]
                win_i = i[idx - self.window_size + 1: idx + 1]
                win_t = t[idx - self.window_size + 1: idx + 1]
                feat = np.stack([win_v, win_i, win_t], axis=1)  # (W,3)
                seqs.append(feat)
                labels.append(y[idx])

        self.seqs = np.stack(seqs).astype(np.float32)
        self.labels = np.array(labels).astype(np.float32)
        print(f"[DATA] 多文件总样本: {self.seqs.shape[0]} 个, 窗长={self.window_size}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


def split_indices(idxs, seed=42):
    # np.random.seed(seed)  # 保证可复现
    idxs = np.array(idxs)
    N = len(idxs)

    # 打乱顺序
    source_idxs = idxs
    idxs = np.random.permutation(idxs)

    # 划分85%和15%
    split_85 = int(N * 0.85)
    idxs_85 = idxs[:split_85]
    idxs_15 = source_idxs[split_85:]  # 测试集

    # 再把85%划分成8:2
    split_train = int(len(idxs_85) * 0.8)
    train_idxs = idxs_85[:split_train]
    val_idxs = idxs_85[split_train:]

    test_idxs = idxs_15

    return train_idxs, val_idxs, test_idxs



# -------------------- 训练函数 --------------------
def train_calce_dnn(excel_paths, dataset_name, work_dir='./work_dir/',
                    batch_size=256, lr=1e-3, epochs=50, window_size=WINDOW_SIZE, patience=10):
    # excel_paths 可能是 str 或 list[str]
    if isinstance(excel_paths, (list, tuple)):
        dataset = CALCESOCDatasetMulti(excel_paths, window_size, use_zscore=True)
    else:
        # 兼容旧用法
        dataset = CALCESOCDataset(excel_paths, window_size, start_row=DATASET_CONFIG.get(dataset_name, 1), use_zscore=True)

    N = len(dataset)
    idxs = np.arange(N)   # 保持原始顺序（用于确定最后15%）

    train_idx, val_idx, test_idx = split_indices(idxs)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    device = torch.device(DEVICE)

    # -------- 模型定义 --------
    model = CNNGRUAttention(in_feats=3, cnn_channels=32, kernel_size=3,
                            pool_size=1, gru_hidden=64, out_feats=1).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.HuberLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    best_loss, patience_counter = float('inf'), 0
    os.makedirs(work_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ---- Train ----
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

        # ---- Val ----
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
            torch.save(model.state_dict(), f"{work_dir}/cnn_gru_attention_soc.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"训练结束，最佳 val_loss={best_loss:.6f}")
    test_and_predict(model, test_loader, device, work_dir=work_dir)


# -------------------- 测试函数 --------------------
def test_and_predict(model, test_loader, device, work_dir='./work_dir/'):
    model.eval()
    y_true, y_pred, indices = [], [], []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            # 注意：这里 Subset 会保留原始索引
            idx = test_loader.dataset.indices[batch_idx * test_loader.batch_size : batch_idx * test_loader.batch_size + len(y)]
            indices.extend(idx)

            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(-1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # ---- 排序还原 ----
    indices = np.array(indices)
    y_true = np.array(y_true)[np.argsort(indices)]
    y_pred = np.array(y_pred)[np.argsort(indices)]

    # ---- 计算指标 ----
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"测试集: MAE={mae:.6f}, MSE={mse:.6f}, RMSE={rmse:.6f}")

    # ---- 绘图 ----
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Real", color='blue')
    plt.plot(y_pred, label="Pred", color='red', linestyle='--')
    plt.xlabel('time')
    plt.ylabel('SOC')
    plt.title('Real vs Pred')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{work_dir}/prediction_vs_true.png")
    plt.show()

    return mae, mse, rmse



# -------------------- 主入口 --------------------
if __name__ == "__main__":
    csv_files = [
        './dataset/csv/DST-0C-50SOC.csv',
        './dataset/csv/DST-0C-80SOC.csv',
    ]
    # dataset_name 仅用于日志/保存名，不再用于 start_row（上面已按文件名各自处理）
    train_calce_dnn(csv_files, dataset_name='DST-0C-MIX', work_dir='./work_dir_new')
