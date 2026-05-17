import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import build_dataloaders_from_cs2_df
from config_cs2 import (
    DEVICE, WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS,
    EPOCHS, LR, PATIENCE, MODELS_DIR
)
from model import TCNRegressor, TimeSeriesModel, TwoBranchFramework, MLPModel, CNNModel

def seed_everything(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    if len(y_true) == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    return rmse, mae

def _epoch(model, loader: DataLoader, optimizer=None, device="cpu", desc=""):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total, n = 0.0, 0
    ys, yps, times_all, cids_all = [], [], [], []
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        tb = None
        cids = None
        if isinstance(batch, (list, tuple)):
            if len(batch) == 4:
                xb, yb, tb, cids = batch
            elif len(batch) == 3:
                xb, yb, third = batch
                try:
                    if torch.is_floating_point(third):
                        tb = third
                    else:
                        cids = third
                except Exception:
                    cids = third
            else:
                xb, yb = batch
        else:
            xb, yb = batch

        xb, yb = xb.to(device), yb.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        yhat = model(xb)
        yhat = torch.sigmoid(yhat)
        loss = F.mse_loss(yhat, yb)

        if train_mode:
            loss.backward()
            optimizer.step()

        total += float(loss.item()) * yb.size(0); n += yb.size(0)
        ys.append(yb.detach().cpu().numpy().reshape(-1))
        yps.append(yhat.detach().cpu().numpy().reshape(-1))
        if tb is not None:
            try:
                times_all.append(tb.detach().cpu().numpy().reshape(-1))
            except Exception:
                times_all.append(np.asarray(tb).reshape(-1))
        if cids is not None:
            try:
                cids_all.append(cids.detach().cpu().numpy().reshape(-1))
            except Exception:
                cids_all.append(np.asarray(cids).reshape(-1))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg = total / max(1, n)
    y  = np.concatenate(ys)  if ys  else np.array([])
    yp = np.concatenate(yps) if yps else np.array([])
    times_flat = np.concatenate(times_all) if times_all else None
    cids_flat  = np.concatenate(cids_all)  if cids_all  else None
    return avg, y, yp, times_flat, cids_flat

def train_tcn_on_table(df, device=DEVICE, run_id=1, fname=None):
    ltr, lva, lte = build_dataloaders_from_cs2_df(df, WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS)
    # TSModel: nasa_lstm_results.xlsx
    
    # TCN
    # model = TCNRegressor(in_dim=3, hidden_size=64, dropout=0.2).to(device)   # TCN

    # LSTM
    # model = TimeSeriesModel(input_dim=3, return_feat=False).to(device) # LSTM

    # MLP
    # model = MLPModel(input_dim=3, seq_len=64).to(device)

    # CNN
    # model = CNNModel(input_dim=3).to(device)

    # Ours
    model = TwoBranchFramework(WINDOW).to(device)  # Ours


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, EPOCHS + 1):
        tr_loss, _, _, _, _ = _epoch(model, ltr, optimizer, device, desc=f"[TCN] Train ep{ep}")
        va_loss, _, _, _, _ = _epoch(model, lva, None, device, desc=f"[TCN]  Val  ep{ep}")
        print(f"[TCN] ep {ep:02d}: train {tr_loss:.6f} | val {va_loss:.6f}")
        if va_loss < best_val - 1e-12:
            best_val = va_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print("[TCN] Early stop.")
                break
    
    if fname is None:
        root = os.path.join(MODELS_DIR, f"run_{run_id}")
    else:
        root = os.path.join(MODELS_DIR, f"run_{run_id}", os.path.splitext(fname)[0])
    os.makedirs(root, exist_ok=True)
    
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model, os.path.join(root, 'best_model.pth'))

    _, y_true, y_pred, times, cids = _epoch(model, lte, None, device, desc="[TCN]  Test")
    return y_true, y_pred, times, cids

