'''
训练kalmannet
'''
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    SLICE_LEN, EPOCHS_WARMUP, EPOCHS_FINETUNE,
    DEVICE, RANDOM_SEED, OUTPUT_DIR, DNN_CHECKPOINT, WEIGHT_DECAY
)
from data_loader import load_all_trajectories
from features import compute_kalman_priors, compute_F1_F2_F4
from model import KalmanNetNet1, CNNGRUAttention
from kalman_filter import KalmanFilterWrapperDNN
from soc_pred import WINDOW_SIZE
from utils import pad_sequence, pad_labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_dnn_pred(feats_full, dnn_model, device):
    T = feats_full.shape[0]
    W = WINDOW_SIZE
    if T < W:
        feats_full = np.vstack([feats_full,
                                np.repeat(feats_full[-1][None], W - T, axis=0)])
        T = W
    windows = np.stack([feats_full[i:i+W] for i in range(T-W+1)],
                       axis=0).astype(np.float32)
    with torch.no_grad():
        inp = torch.from_numpy(windows).to(device)
        y_pred_win = dnn_model(inp).cpu().numpy().squeeze(-1)
    soc_pred = np.empty(T, dtype=np.float32)
    soc_pred[:W-1] = y_pred_win[0]
    soc_pred[W-1:] = y_pred_win
    return soc_pred


def train():
    device = torch.device(DEVICE)
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trajectories = load_all_trajectories()

    dnn_model = CNNGRUAttention(in_feats=2, cnn_channels=32,
                                kernel_size=3, pool_size=1,
                                gru_hidden=64, out_feats=1).to(device)
    ckpt = torch.load(DNN_CHECKPOINT, map_location=device)
    dnn_model.load_state_dict(
        ckpt.get('model_state_dict', ckpt)
        if isinstance(ckpt, dict) else ckpt
    )
    # dnn_model是不需要训的
    dnn_model.eval()

    kf_model = KalmanNetNet1().to(device)
    kf_wrapper = KalmanFilterWrapperDNN(kf_model)
    kf_wrapper.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        kf_model.parameters(),
        lr=1e-5,
        weight_decay=WEIGHT_DECAY
    )

    pre_X, pre_V, pre_I, pre_dt, pre_C, pre_Ytrue = [], [], [], [], [], []
    for traj in trajectories:
        v_norm = (traj['voltage'] - traj['voltage'].min()) / \
                 (traj['voltage'].max() - traj['voltage'].min())
        is_charge = traj['id'].lower().startswith('charge')
        soc_full = traj['soc']
        soc_true = soc_full[WINDOW_SIZE - 1:]
        q_cum_full = soc_full if is_charge else (1.0 - soc_full)
        feats = np.stack([v_norm, q_cum_full], axis=1).astype(np.float32)

        soc_pred = compute_dnn_pred(feats, dnn_model, device)

        traj_pred = {"soc": soc_pred}
        kalman_prior = compute_kalman_priors(traj_pred)
        F1, F2, F4 = compute_F1_F2_F4(traj_pred, kalman_prior)
        X_full = np.concatenate([F1, F2, F4], axis=1)[WINDOW_SIZE-1:]

        t = traj['time'][WINDOW_SIZE-1:]
        raw_dt = np.diff(t, prepend=t[0])
        pos = raw_dt[raw_dt > 0]
        median_dt = float(np.median(pos)) if pos.size else 1.0
        dt_arr = np.where(raw_dt>1e-3, raw_dt, median_dt)

        pre_X.append(X_full)
        pre_Ytrue.append(soc_true)
        pre_V.append(traj['voltage'][WINDOW_SIZE-1:])
        pre_I.append(traj['current'][WINDOW_SIZE-1:])
        pre_dt.append(dt_arr)
        pre_C.append(traj['capacity'])

    # ---- V2 Warm-up ----
    print("\n==== V2 Warm-up ====")
    for epoch in range(EPOCHS_WARMUP):
        kf_model.train()
        losses = []
        for traj_idx, (X_diff, Ytrue, Vobs, Iobs, Dtobs, Cap) in enumerate(
                zip(pre_X, pre_Ytrue, pre_V, pre_I, pre_dt, pre_C)):
            T = X_diff.shape[0]
            for start in range(0, T, SLICE_LEN):
                end = min(start+SLICE_LEN, T)
                if end-start<=1: continue

                x = torch.tensor(X_diff[start:end][None],
                                 dtype=torch.float32, device=device)
                vt = torch.tensor(Vobs[start:end][None],
                                  dtype=torch.float32, device=device)
                I  = torch.tensor(Iobs[start:end][None],
                                  dtype=torch.float32, device=device)
                dt = torch.tensor(Dtobs[start:end][None],
                                  dtype=torch.float32, device=device)
                cap = torch.tensor([[Cap]],
                                   dtype=torch.float32, device=device)
                y  = torch.tensor(Ytrue[start:end][None],
                                  dtype=torch.float32, device=device)

                optimizer.zero_grad()
                x_est, _ = kf_wrapper(x, vt, I, dt, cap)
                loss = criterion(x_est.squeeze(-1), y)

                if torch.isnan(loss):
                    print(f"[ERROR] NaN loss at warm-up epoch={epoch+1}, traj={traj_idx}, slice={start}:{end}")
                    raise RuntimeError("Warm-up loss is NaN")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(kf_model.parameters(), max_norm=0.1)
                optimizer.step()
                losses.append(loss.detach())

        mean_loss = torch.stack(losses).mean().item()
        print(f"Epoch {epoch+1}/{EPOCHS_WARMUP}, Loss: {mean_loss:.6e}")

    for name, p in kf_model.named_parameters():
        if torch.isnan(p).any():
            raise RuntimeError(f"[CRITICAL] 参数{name} 包含 NaN，中止训练")

    torch.save(kf_model.state_dict(), os.path.join(OUTPUT_DIR, 'warmup.pth'))

    for g in optimizer.param_groups:
        g['lr'] = 5e-6

    # ---- V1 Fine-tune ----
    print("\n==== V1 Fine-tune ====")
    X_pad  = pad_sequence(pre_X,  pad_value=0.0)
    Y_pad  = pad_labels(pre_Ytrue, pad_value=0.0)
    M_pad  = pad_labels([np.ones_like(y) for y in pre_Ytrue], pad_value=0.0)
    V_pad  = pad_labels(pre_V,  pad_value=0.0)
    I_pad  = pad_labels(pre_I,  pad_value=0.0)
    Dt_pad = pad_labels(pre_dt, pad_value=0.0)

    max_len = max(len(y) for y in pre_Ytrue)
    C_pad = np.array([[c] * max_len for c in pre_C], dtype=np.float32)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_pad, dtype=torch.float32, device=device),
        torch.tensor(Y_pad, dtype=torch.float32, device=device),
        torch.tensor(M_pad, dtype=torch.float32, device=device),
        torch.tensor(V_pad, dtype=torch.float32, device=device),
        torch.tensor(I_pad, dtype=torch.float32, device=device),
        torch.tensor(Dt_pad, dtype=torch.float32, device=device),
        torch.tensor(C_pad, dtype=torch.float32, device=device),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(EPOCHS_FINETUNE):
        kf_model.train()
        losses = []
        for Xb, Yb, Mb, Vb, Ib, Dtb, Cb in loader:
            optimizer.zero_grad()
            segs = (Yb.size(1) + SLICE_LEN - 1)//SLICE_LEN
            for i in range(segs):
                s, e = i*SLICE_LEN, min((i+1)*SLICE_LEN, Yb.size(1))
                if e-s<=1: continue

                xseg  = Xb[:,s:e,:]
                yseg  = Yb[:,s:e]
                mseg  = Mb[:,s:e]
                vtseg = Vb[:,s:e]
                Iseg  = Ib[:,s:e]
                dtseg = Dtb[:,s:e]
                cap_t = Cb[:,s].view(-1,1)

                est, _ = kf_wrapper(xseg, vtseg, Iseg, dtseg, cap_t)
                mask = mseg.sum().item()
                if mask==0: continue

                loss = ((est.squeeze(-1)-yseg)**2 * mseg).sum()/mask
                if torch.isnan(loss):
                    print(f"[ERROR] NaN loss at fine-tune epoch={epoch+1}, slice={s}:{e}")
                    raise RuntimeError("Fine-tune loss is NaN → 检查数据/超参")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(kf_model.parameters(), max_norm=0.1)
                optimizer.step()
                losses.append(loss.detach())

        mean_loss = torch.stack(losses).mean().item()
        print(f"Epoch {epoch+1}/{EPOCHS_FINETUNE}, Loss: {mean_loss:.6e}")

    torch.save(kf_model.state_dict(), os.path.join(OUTPUT_DIR, 'final.pth'))


if __name__ == '__main__':
    train()
