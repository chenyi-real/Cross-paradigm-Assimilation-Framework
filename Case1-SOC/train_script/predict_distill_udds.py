#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_distill_udds.py

极简：加载已训练好的 distill 模型，对 A/B/C/D (DATASET_PARTS) 做测试，
并把 true/pred 保存为 true_label.npy / pred_label.npy 到 RESULTS_DIR 下。

依赖：
- configs/config_distill_udds.py 里提供：
  DATA_DIR, CSV_FILES, DEVICE, DATASET_PARTS, MODELS_DIR, RESULTS_DIR (新增)
- dataloader.load_UDDS_csv_data
- train_distill_udds.test_dstill_on_table  (会自动加载 best_model_distill.pth / best_student_distill.pth)
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

from configs.config_distill_udds import (
    DATA_DIR, CSV_FILES, DEVICE, DATASET_PARTS, RESULTS_DIR
)
from dataloader import load_UDDS_csv_data
from train_distill_udds import test_dstill_on_table, compute_metrics


def _split_for_part(part: str, all_csv_folders):
    """
    与 main_distill_udds.py 保持一致的 test 切分逻辑。
    all_csv_folders: e.g. ['Cycling_1', 'Cycling_2', ...]
    """
    val_csv = ['Cycling_1', 'Cycling_2']

    if part == 'A':
        test_csv = ['Cycling_3']
    elif part == 'B':
        test_csv = ['Cycling_4']
    elif part == 'C':
        test_csv = [f"Cycling_{i}" for i in range(8, 12)]
    else:
        # 默认 D
        test_csv = [f"Cycling_{i}" for i in range(12, 15)]

    train_csv = [x for x in all_csv_folders if x not in val_csv and x not in test_csv]
    return train_csv, val_csv, test_csv


def _save_arrays(out_dir: str, y_true, y_pred, times=None, cids=None):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "true_label.npy"), np.asarray(y_true, dtype=np.float32))
    np.save(os.path.join(out_dir, "pred_label.npy"), np.asarray(y_pred, dtype=np.float32))
    if times is not None:
        np.save(os.path.join(out_dir, "times.npy"), np.asarray(times))
    if cids is not None:
        np.save(os.path.join(out_dir, "cids.npy"), np.asarray(cids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=int, default=1, help="加载哪个 run 的模型权重（MODELS_DIR/run_{run_id}/...）")
    ap.add_argument("--parts", type=str, default=",".join(DATASET_PARTS),
                    help="要测试哪些 part，用逗号分隔，例如 'A,B'；默认使用 config.DATASET_PARTS")
    args = ap.parse_args()

    run_id = int(args.run_id)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]

    print(f"[Device] {DEVICE}")
    print(f"[Predict] run_id={run_id} parts={parts}")
    print(f"[Save] RESULTS_DIR={RESULTS_DIR}")

    for part in parts:
        _, _, test_csv = _split_for_part(part, CSV_FILES)

        print(f"\n========== [Predict] part={part} ==========")
        print(f"[Data] test folders = {test_csv}")

        test_df = load_UDDS_csv_data(DATA_DIR, test_csv, part, split="test")

        # 关键：只测试，且会从 MODELS_DIR/run_{run_id}/{part}/ 下加载 teacher+student 权重
        y_true, y_pred, times, cids = test_dstill_on_table(
            test_df, device=DEVICE, run_id=run_id, fname=part
        )

        rmse, mae = compute_metrics(y_true, y_pred)
        print(f"[Metrics] part={part}  RMSE={rmse:.6f}  MAE={mae:.6f}")

        out_dir = os.path.join(RESULTS_DIR, f"run_{run_id}", str(part))
        _save_arrays(out_dir, y_true, y_pred, times=times, cids=cids)
        print(f"[Saved] {out_dir}/true_label.npy, pred_label.npy (plus times.npy, cids.npy)")

    print("\n[Done] Predict finished.")


if __name__ == "__main__":
    main()
