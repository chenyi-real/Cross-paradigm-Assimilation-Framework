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
from dataloader import  load_UDDS_csv_data

from configs import config_transfer_udds
from configs.config_transfer_udds import DATA_DIR, CSV_FILES, DEVICE, RESULTS_XLSX, PLOTS_DIR, PLOT, DATASET_PARTS, RUNS
from dataloader import  load_UDDS_csv_data
from train_transfer_udds import seed_everything, train_tcn_on_table, compute_metrics


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


def _one_experiment(run_id: int, per_file_results: dict):
    table_rmses = []
    table_maes = []

    base_seed = 20251010
    seed_everything(base_seed + int(run_id))


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

        # 在这里读取好 train_df, val_df, test_df.
        # print(f"\n[Data] {fpath}")
        # df = load_one_csv(fpath)

        
        train_df = load_UDDS_csv_data(DATA_DIR, train_csv, fname, split='train')
        val_df   = load_UDDS_csv_data(DATA_DIR, val_csv, fname, split='val')
        test_df  = load_UDDS_csv_data(DATA_DIR, test_csv, fname, split='test')
        
        print("[Stage] Train + Test")
        y_true, y_pred, times, cids = train_tcn_on_table(train_df, val_df, test_df, device=DEVICE, run_id=run_id, fname=fname)

        rmse, mae = compute_metrics(y_true, y_pred)
        print(f"[Result][run={run_id}] {fname}  RMSE={rmse:.6f}  MAE={mae:.6f}")
        table_rmses.append(rmse)
        table_maes.append(mae)

        # ====== 收集每个CSV文件的结果 ======
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

    os.makedirs(config_transfer_udds.EXP_DIR, exist_ok=True)
    cfg_path = os.path.join(config_transfer_udds.EXP_DIR, "config.yaml")

    with open(cfg_path, "w", encoding="utf-8") as f:
        for k, v in vars(config_transfer_udds).items():
            # 跳过内置变量和模块等
            if k.startswith("_") or inspect.ismodule(v) or inspect.isfunction(v):
                continue
            f.write(f"{k}: {v}\n")

    print(f"[Config] Saved to {cfg_path}")



    print(f"[Device] {DEVICE}")
    start_run = 1
    per_file_results = {}  # 存储每个CSV的10次实验结果

    for run_id in range(start_run, RUNS + 1):
        print(f"\n========== [Experiment run {run_id}/10] ==========")
        rmse_avg, mae_avg = _one_experiment(run_id, per_file_results)
        _write_total_result(RESULTS_XLSX, run_id, rmse_avg, mae_avg)
        print(f"[Export] Appended run={run_id} to {RESULTS_XLSX}")

    # ====== 将每个CSV文件的10次结果保存为单独的xlsx ======
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
        print(f"[Export] Per-file 10-run results saved: {save_path}")


if __name__ == "__main__":
    main()
