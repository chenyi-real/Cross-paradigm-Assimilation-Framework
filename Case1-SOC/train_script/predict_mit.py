import os
import sys
import json

# 保证能 import 到项目内模块
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch

from dataloader import build_dataloaders_from_mit_df, load_MIT_csv_data
from train import _epoch, seed_everything

from configs.config_mit import (
    DEVICE, WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS, MODELS_DIR
)
from configs.config_mit import DATA_DIR, CSV_FILES

RESULTS_DIR = './test_results_20250311/baseline/MIT/'


def extract_lte_dataset_to_dataframe(lte):
    """
    将 DataLoader 中的 dataset 提取出来，保存成 DataFrame。
    """
    dataset = lte.dataset
    rows = []

    for i in range(len(dataset)):
        item = dataset[i]

        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                x = item[0]
                y = item[1]
            else:
                x = item[0]
                y = None
        else:
            x = item
            y = None

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        else:
            x = np.array(x)

        if y is not None:
            if torch.is_tensor(y):
                y = y.detach().cpu().numpy()
            else:
                y = np.array(y)

        row = {
            "sample_idx": i,
            "x_shape": str(tuple(x.shape)),
            "x_json": json.dumps(x.tolist(), ensure_ascii=False)
        }

        if y is not None:
            if np.ndim(y) == 0:
                row["y_true"] = float(y)
            elif np.size(y) == 1:
                row["y_true"] = float(np.reshape(y, -1)[0])
            else:
                row["y_true_json"] = json.dumps(y.tolist(), ensure_ascii=False)

        rows.append(row)

    df_windows = pd.DataFrame(rows)
    return df_windows


def build_row_level_pred_df(df_raw, y_true, y_pred, window, stride):
    """
    按 (Date, cycle) 分段对齐窗口预测
    """
    df_out = df_raw.copy().reset_index(drop=True)

    df_out["true_window"] = np.nan
    df_out["pred_window"] = np.nan
    df_out["window_id"] = np.nan

    pred_idx = 0

    for cycle_id, df_cycle in df_out.groupby(["batch_date", "cycle"]):

        cycle_indices = df_cycle.index.tolist()
        cycle_len = len(cycle_indices)

        num_windows = (cycle_len - window) // stride + 1
        if num_windows <= 0:
            continue

        for w in range(num_windows):

            end_pos = w * stride + window - 1

            if end_pos >= cycle_len:
                break

            global_idx = cycle_indices[end_pos]

            df_out.loc[global_idx, "window_id"] = pred_idx
            df_out.loc[global_idx, "pred_window"] = float(
                np.asarray(y_pred[pred_idx]).reshape(-1)[0]
            )
            df_out.loc[global_idx, "true_window"] = float(
                np.asarray(y_true[pred_idx]).reshape(-1)[0]
            )

            pred_idx += 1

    return df_out


def build_window_level_pred_df(y_true, y_pred, window, stride, total_rows):

    rows = []
    n_samples = len(y_pred)

    for i in range(n_samples):

        start_idx = i * stride
        end_idx = start_idx + window - 1

        row = {
            "window_id": i,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "window": window,
            "stride": stride,
            "y_true": float(np.asarray(y_true[i]).reshape(-1)[0]),
            "y_pred": float(np.asarray(y_pred[i]).reshape(-1)[0]),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def test_one_csv(fname: str, run_id: int = 1):
    """
    使用已经训练好的 best_model.pth 对单个 CSV 做测试
    """

    base = os.path.splitext(fname)[0]

    # -------- 1. 加载原始数据 --------
    fpath = os.path.join(DATA_DIR, fname)
    df = load_MIT_csv_data(fpath)

    _, _, lte, df_te = build_dataloaders_from_mit_df(
        df, WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS
    )

    # -------- 2. 加载模型 --------
    model_path = os.path.join(
        MODELS_DIR, f"run_{run_id}", base, "best_model.pth"
    )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.eval()

    # -------- 3. 测试 --------
    _, y_true, y_pred, _, _ = _epoch(
        model, lte, optimizer=None, device=DEVICE, desc=f"[Test] {base}"
    )

    return df_te, lte, y_true, y_pred


def main():

    seed_everything(20251010)

    run_id = 1
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"[Test] run_id = {run_id}")
    print(f"[Save] RESULTS_DIR = {RESULTS_DIR}")

    for fname in CSV_FILES:

        base = os.path.splitext(fname)[0]
        print(f"\n========== Testing {base} ==========")

        df_raw, lte, y_true, y_pred = test_one_csv(fname, run_id)

        # -------- 保存结果 --------
        out_dir = os.path.join(RESULTS_DIR, base)
        os.makedirs(out_dir, exist_ok=True)

        # npy
        np.save(os.path.join(out_dir, "true_label.npy"), y_true)
        np.save(os.path.join(out_dir, "pred_label.npy"), y_pred)

        # csv
        df_raw_with_pred = build_row_level_pred_df(
            df_raw=df_raw,
            y_true=y_true,
            y_pred=y_pred,
            window=WINDOW,
            stride=STRIDE
        )

        df_raw_with_pred.to_csv(
            os.path.join(out_dir, "raw_with_pred.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        print(f"[Saved] {base}")
        print(f"  true_label.npy: {y_true.shape}")
        print(f"  pred_label.npy: {y_pred.shape}")
        print(f"  raw_with_pred.csv")

    print("\n[Test Finished] All CSV_FILES tested successfully.")


if __name__ == "__main__":
    main()