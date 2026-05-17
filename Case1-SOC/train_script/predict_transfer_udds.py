#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_transfer_udds.py
- Load trained (fine-tuned) model once (one run) and evaluate on UDDS transfer splits.
- Save predictions / labels to RESULTS_DIR:
    {RESULTS_DIR}/run_{run_id}/{part}/pred_label.npy
    {RESULTS_DIR}/run_{run_id}/{part}/true_label.npy
    {RESULTS_DIR}/run_{run_id}/{part}/times.npy
    {RESULTS_DIR}/run_{run_id}/{part}/cids.npy
  plus a summary.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 让脚本从项目根目录 import（沿用你 main_transfer_udds.py 的写法）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataloader import load_UDDS_csv_data, build_dataloaders_from_udds_df
from model import TCNRegressor, TimeSeriesModel, TwoBranchFramework, MLPModel, CNNModel

from configs.config_transfer_udds import (
    DATA_DIR, CSV_FILES, DEVICE,
    WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS,
    MODELS_DIR, METHOD, EXP_DIR, DATASET_PARTS,
)

# 你要求新增的 RESULTS_DIR：若 config 里没有，就 fallback
try:
    from configs.config_transfer_udds import RESULTS_DIR  # type: ignore
except Exception:
    RESULTS_DIR = os.path.join(EXP_DIR, "predict_results")


@torch.no_grad()
def _epoch_test(model, loader: DataLoader, device="cpu", desc="Test"):
    model.eval()
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
                # third 可能是 times 或 cids
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

        xb = xb.to(device)
        yb = yb.to(device)

        yhat = model(xb)
        yhat = torch.sigmoid(yhat)

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

    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(yps) if yps else np.array([])
    times = np.concatenate(times_all) if times_all else None
    cids = np.concatenate(cids_all) if cids_all else None
    return y_true, y_pred, times, cids


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    if len(y_true) == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return rmse, mae


def build_model(method: str, device: str):
    method = str(method)

    if method == "Ours":
        model = TwoBranchFramework(exp_dir=EXP_DIR).to(device)
    elif method == "MLP":
        model = MLPModel(input_dim=3, seq_len=WINDOW).to(device)
    elif method == "TCN":
        model = TCNRegressor(in_dim=3, hidden_size=64, dropout=0.2).to(device)
    elif method == "LSTM":
        model = TimeSeriesModel(input_dim=3, return_feat=False).to(device)
    elif method == "CNN":
        model = CNNModel(input_dim=3).to(device)
    else:
        raise ValueError(f"Unknown METHOD={method}")
    return model


def get_split_for_part(part: str):
    """
    完全复刻你 main_transfer_udds.py 中对 A/B/C/D 的划分逻辑
    """
    part = str(part).upper()
    val_csv = ["Cycling_1", "Cycling_2"]

    if part == "A":
        test_csv = ["Cycling_3"]
    elif part == "B":
        test_csv = ["Cycling_4"]
    elif part == "C":
        test_csv = [f"Cycling_{i}" for i in range(8, 12)]
    else:
        test_csv = [f"Cycling_{i}" for i in range(12, 15)]

    train_csv = [x for x in CSV_FILES if x not in val_csv and x not in test_csv]
    return train_csv, val_csv, test_csv


def load_finetuned_weights(model, run_id: int, part: str, device: str):
    """
    对应 train_transfer_udds.py 的保存路径：best_model_fc_finetuned.pth
    """
    ckpt = os.path.join(MODELS_DIR, f"run_{run_id}", str(part), "best_model_fc_finetuned.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt}")

    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # 允许 strict=False，防止你模型内部有轻微key差异
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[Warn] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    print(f"[Load] run={run_id} part={part} <- {ckpt}")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=int, default=1, help="只测试某一次训练（默认1）")
    ap.add_argument("--parts", type=str, default=",".join(list(DATASET_PARTS)),
                    help="要测试的 part，逗号分隔，例如 A,B,C,D")
    ap.add_argument("--results_dir", type=str, default=RESULTS_DIR, help="保存目录（默认取 config.RESULTS_DIR）")
    args = ap.parse_args()

    device = DEVICE
    run_id = int(args.run_id)
    parts = [p.strip().upper() for p in str(args.parts).split(",") if p.strip()]

    out_root = os.path.join(args.results_dir, f"run_{run_id}")
    os.makedirs(out_root, exist_ok=True)

    summary_rows = []

    for part in parts:
        print(f"\n========== [Predict] run={run_id} part={part} ==========")

        # 构造 test_df（拆分逻辑同 main_transfer_udds.py）
        train_csv, val_csv, test_csv = get_split_for_part(part)
        test_df = load_UDDS_csv_data(DATA_DIR, test_csv, part, split="test")

        # 用现成的 build_dataloaders_from_udds_df 生成 lte（train/val 这里传 dummy，不会用于预测）
        dummy_df = test_df.iloc[: max(1, min(len(test_df), 256))].copy()
        _, _, lte = build_dataloaders_from_udds_df(
            dummy_df, dummy_df, test_df,
            WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS
        )

        # 建模 + 加载权重
        model = build_model(METHOD, device=device)
        ckpt_path = load_finetuned_weights(model, run_id=run_id, part=part, device=device)

        # 推理
        y_true, y_pred, times, cids = _epoch_test(model, lte, device=device, desc=f"[{part}] Test")

        rmse, mae = compute_metrics(y_true, y_pred)
        print(f"[Metrics] part={part}  RMSE={rmse:.6f}  MAE={mae:.6f}")

        # 保存 npy
        part_dir = os.path.join(out_root, part)
        os.makedirs(part_dir, exist_ok=True)

        np.save(os.path.join(part_dir, "true_label.npy"), y_true)
        np.save(os.path.join(part_dir, "pred_label.npy"), y_pred)
        if times is not None:
            np.save(os.path.join(part_dir, "times.npy"), times)
        if cids is not None:
            np.save(os.path.join(part_dir, "cids.npy"), cids)

        summary_rows.append({
            "run": run_id,
            "part": part,
            "method": METHOD,
            "checkpoint": ckpt_path,
            "test_csv": ",".join(test_csv),
            "RMSE": rmse,
            "MAE": mae,
            "n_points": int(len(y_true)),
        })

        print(f"[Save] {part_dir}/(true_label.npy, pred_label.npy, times.npy, cids.npy)")

    # 汇总保存
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_root, "summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[Done] Summary saved: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
