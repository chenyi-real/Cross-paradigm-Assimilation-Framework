import os
import sys
# 获取 SOC_TwoBranch 作为项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect

import configs.config_distill_udds as config_distill_udds
from configs.config_distill_udds import (
    DATA_DIR, CSV_FILES, DEVICE, RESULTS_XLSX, PLOTS_DIR,
    PLOT, DATASET_PARTS, RUNS
)
from dataloader import load_UDDS_csv_data
from train_distill_udds import (
    seed_everything,
    train_dstill_on_table,
    compute_metrics,
    test_dstill_on_table,
)


def _save_truth_pred_plot(times, y_true, y_pred, *, out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(times, y_true, label="True")
    plt.plot(times, y_pred, label="Pred")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("SOC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _get_next_run(xlsx_path: str):
    try:
        df = pd.read_excel(xlsx_path)
        if "run" in df.columns and len(df) > 0:
            max_run = int(pd.to_numeric(df["run"], errors="coerce").fillna(0).max())
            return max_run + 1
        return 1
    except Exception:
        return 1


def _get_existing_runs(xlsx_path: str):
    """
    从 RESULTS_XLSX 中读取已经存在的 run 编号，返回一个 set[int]。
    """
    try:
        df = pd.read_excel(xlsx_path)
        if "run" not in df.columns:
            return set()
        runs = pd.to_numeric(df["run"], errors="coerce")
        runs = runs[~runs.isna()].astype(int)
        return set(runs.tolist())
    except Exception:
        return set()


def _write_total_result(xlsx_path: str, run_id: int, rmse_avg: float, mae_avg: float):
    cols = ["run", "RMSE", "MAE"]
    try:
        df = pd.read_excel(xlsx_path)
        for c in cols:
            if c not in df.columns:
                df[c] = None
    except Exception:
        df = pd.DataFrame(columns=cols)

    df.loc[len(df)] = {"run": int(run_id), "RMSE": rmse_avg, "MAE": mae_avg}
    os.makedirs(os.path.dirname(xlsx_path) or ".", exist_ok=True)
    df.to_excel(xlsx_path, index=False)


def _one_experiment(run_id: int, per_file_results: dict, *, only_test: bool):
    """
    only_test = False: 训练 + 测试
    only_test = True : 只测试（假设模型已经训练好并保存）
    """
    table_rmses = []
    table_maes = []

    base_seed = 20251010
    # 只测试时通常不需要重新设置随机种子，不过保留也无妨
    seed_everything(base_seed + int(run_id))

    mode_str = "TEST ONLY" if only_test else "TRAIN + TEST"
    print(f"[Mode] run={run_id} -> {mode_str}")

    for fname in DATASET_PARTS:
        if fname == 'A':
            val_csv = ['Cycling_1', 'Cycling_2']
            test_csv = ['Cycling_3']
            train_csv = [x for x in CSV_FILES if x not in val_csv and x not in test_csv]

        elif fname == 'B':
            val_csv = ['Cycling_1', 'Cycling_2']
            test_csv = ['Cycling_4']
            train_csv = [x for x in CSV_FILES if x not in val_csv and x not in test_csv]

        elif fname == 'C':
            val_csv = ['Cycling_1', 'Cycling_2']
            test_csv = [f"Cycling_{i}" for i in range(8, 12)]
            train_csv = [x for x in CSV_FILES if x not in val_csv and x not in test_csv]
        else:
            val_csv = ['Cycling_1', 'Cycling_2']
            test_csv = [f"Cycling_{i}" for i in range(12, 15)]
            train_csv = [x for x in CSV_FILES if x not in val_csv and x not in test_csv]

        # ====== 数据加载 ======
        train_df = None
        val_df = None

        if not only_test:
            train_df = load_UDDS_csv_data(DATA_DIR, train_csv, fname, split='train')
            val_df = load_UDDS_csv_data(DATA_DIR, val_csv, fname, split='val')

        test_df = load_UDDS_csv_data(DATA_DIR, test_csv, fname, split='test')

        print("[Stage] Train + Test" if not only_test else "[Stage] Test Only")

        # ====== 训练 + 测试 或 仅测试 ======
        if only_test:
            # 需要你在 train_distill_udds.py 中实现测试函数：
            # y_true, y_pred, times, cids = test_dstill_on_table(test_df, device, run_id, fname)
            y_true, y_pred, times, cids = test_dstill_on_table(
                test_df, device=DEVICE, run_id=run_id, fname=fname
            )
        else:
            y_true, y_pred, times, cids = train_dstill_on_table(
                train_df, val_df, test_df,
                device=DEVICE, run_id=run_id, fname=fname
            )

        rmse, mae = compute_metrics(y_true, y_pred)
        print(f"[Result][run={run_id}] {fname}  RMSE={rmse:.6f}  MAE={mae:.6f}")
        table_rmses.append(rmse)
        table_maes.append(mae)

        # ====== 收集每个 CSV 文件的结果 ======
        base = os.path.splitext(fname)[0]
        if base not in per_file_results:
            per_file_results[base] = []
        per_file_results[base].append({
            "run": run_id,
            "RMSE": rmse,
            "MAE": mae
        })

        # ====== 绘图控制 ======
        if PLOT:
            root = os.path.join(PLOTS_DIR, f"run_{run_id}", base)
            os.makedirs(root, exist_ok=True)
            uniq = np.unique(cids)
            for cid in uniq:
                m = (cids == cid)
                out_path = os.path.join(root, f"cycle_{int(cid)}.png")
                _save_truth_pred_plot(
                    times[m], y_true[m], y_pred[m],
                    out_path=out_path,
                    title=f"{fname} cycle {int(cid)} (run {run_id})"
                )
            print(f"[Plot] Saved {len(uniq)} plots to {root}")
        else:
            print(f"[Plot] Skipped (PLOT={PLOT})")

    rmse_avg = float(np.nanmean(table_rmses)) if table_rmses else float("nan")
    mae_avg = float(np.nanmean(table_maes)) if table_maes else float("nan")
    print(f"\n[Summary][run={run_id}] avg RMSE={rmse_avg:.6f}  MAE={mae_avg:.6f}")
    return rmse_avg, mae_avg


def main():
    os.makedirs(config_distill_udds.EXP_DIR, exist_ok=True)
    cfg_path = os.path.join(config_distill_udds.EXP_DIR, "config.yaml")

    # 保存当前 config
    with open(cfg_path, "w", encoding="utf-8") as f:
        for k, v in vars(config_distill_udds).items():
            # 跳过内置变量和模块等
            if k.startswith("_") or inspect.ismodule(v) or inspect.isfunction(v):
                continue
            f.write(f"{k}: {v}\n")

    print(f"[Config] Saved to {cfg_path}")
    print(f"[Device] {DEVICE}")

    # 已有的 run 记录
    existing_runs = _get_existing_runs(RESULTS_XLSX)
    print(f"[Info] Existing runs in {RESULTS_XLSX}: {sorted(existing_runs) if existing_runs else 'None'}")

    # 若 config 中没有该字段，则默认 False
    RETEST_COMPLETED_RUNS = getattr(config_distill_udds, "RETEST_COMPLETED_RUNS", False)
    print(f"[Config] RETEST_COMPLETED_RUNS = {RETEST_COMPLETED_RUNS}")

    start_run = 1
    per_file_results = {}  # 存储每个 CSV 的多次实验结果

    for run_id in range(start_run, RUNS + 1):
        already_done = run_id in existing_runs
        # 若该 run 已存在且不希望重新训练，就只测试
        only_test = already_done and not RETEST_COMPLETED_RUNS

        mode_str = "test-only" if only_test else "train+test"
        print(f"\n========== [Experiment run {run_id}/{RUNS}] ({mode_str}) ==========")

        rmse_avg, mae_avg = _one_experiment(run_id, per_file_results, only_test=only_test)

        # 是否需要往总表里写这一轮的平均结果：
        # - 如果 run 之前不存在：必须写（新 run）
        # - 如果 run 已存在但 RETEST_COMPLETED_RUNS=True：视为新的实验记录，继续追加写入
        # - 如果 run 已存在且 RETEST_COMPLETED_RUNS=False：这一轮只是复测，不再往总表追加，避免重复 run 号
        need_write_summary = (not already_done) or RETEST_COMPLETED_RUNS

        if need_write_summary:
            _write_total_result(RESULTS_XLSX, run_id, rmse_avg, mae_avg)
            print(f"[Export] Appended run={run_id} to {RESULTS_XLSX}")
        else:
            print(f"[Export] Skip writing summary for run={run_id} (already exists & only test).")

    # ====== 将每个 CSV 文件的多次结果保存为单独的 xlsx ======
    result_dir = os.path.dirname(RESULTS_XLSX) or "."
    for base, results in per_file_results.items():
        df = pd.DataFrame(results)
        df.loc[len(df)] = {
            "run": "avg",
            "RMSE": df["RMSE"].mean(),
            "MAE": df["MAE"].mean()
        }
        save_path = os.path.join(result_dir, f"{base}_results.xlsx")
        df.to_excel(save_path, index=False)
        print(f"[Export] Per-file multi-run results saved: {save_path}")


if __name__ == "__main__":
    main()
