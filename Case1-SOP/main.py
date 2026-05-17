import os
import glob, re
import numpy as np
import pandas as pd
import time

from config import (DATA_DIR, DATASET_NAME, NETWORK, RUNS,
                    UDDS_DIR, WINDOW_UDDS, STRIDE_UDDS, BATCH_SIZE, NUM_WORKERS, DISTILL_ENABLE,
                    WHOLE_ENABLE, WHOLE_ARCH, DEVICE, UDDS_GROUP)
from dataloader import load_one_csv, build_udds_dataloaders
from train import set_seed, train_on_nasa, train_on_calce, train_on_mit, compute_metrics, train_whole_on_udds

CSV_FILES = ["B0005.csv", "B0006.csv", "B0007.csv", "B0018.csv"]

def compute_mape(y_true, y_pred, min_mag=1.0):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.abs(y_true) >= min_mag
    if not np.any(mask):
        return np.nan

    mape = np.mean(np.abs(y_pred[mask] - y_true[mask]) / np.abs(y_true[mask]))
    return float(mape)

def _results_xlsx_for_network():
    net = str(NETWORK).upper()
    ds = str(DATASET_NAME).upper()
    return f"./{net}-{ds}_results.xlsx"

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

def _write_total_result_mit(xlsx_path: str, run_id: int, mae_avg: float, mape_avg: float, rmse_avg: float):
    cols = ["run", "MAE", "MAPE", "RMSE"]
    try:
        df = pd.read_excel(xlsx_path)
        for c in cols:
            if c not in df.columns:
                df[c] = None
    except Exception:
        df = pd.DataFrame(columns=cols)
    df.loc[len(df)] = {"run": int(run_id), "MAE": mae_avg, "MAPE":mape_avg, "RMSE": rmse_avg}
    os.makedirs(os.path.dirname(xlsx_path) or ".", exist_ok=True)
    df.to_excel(xlsx_path, index=False)

def _results_for_nasa() -> str:
    net_name = str(NETWORK).lower()
    root = os.path.join(".", "results", "nasa", net_name)
    os.makedirs(root, exist_ok=True)
    return root

def _write_per_file(root_dir: str, fname: str, run_id: int, rmse: float, mae: float):
    base = os.path.splitext(os.path.basename(fname))[0]
    xlsx_path = os.path.join(root_dir, f"{base}.xlsx")

    cols = ["run", "RMSE", "MAE"]
    try:
        df = pd.read_excel(xlsx_path)
        for c in cols:
            if c not in df.columns:
                df[c] = None
    except Exception:
        df = pd.DataFrame(columns=cols)

    df.loc[len(df)] = {"run": int(run_id), "RMSE": float(rmse), "MAE": float(mae)}
    os.makedirs(os.path.dirname(xlsx_path) or ".", exist_ok=True)
    df.to_excel(xlsx_path, index=False)

def _whole_results(arch: str, group: str) -> str:
    root_dir = "results_whole_distill" if DISTILL_ENABLE else "results_whole"
    path = os.path.join(".", root_dir, arch, group, "metrics.xlsx")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def _iter_mit_csvs(root: str):
    def _natkey_from_name(name: str):
        parts = re.split(r'(\d+)', name)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    files = []
    if not os.path.isdir(root):
        return files

    subdirs = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    subdirs = sorted(subdirs, key=_natkey_from_name)

    for sub in subdirs:
        subdir = os.path.join(root, sub)
        cands = [
            p for p in glob.glob(os.path.join(subdir, "*.csv"))
            if "discharge" in os.path.basename(p).lower()
        ]
        cands = sorted(cands, key=lambda p: _natkey_from_name(os.path.basename(p)))
        files.extend(cands)

    return files

def _mit_experiment(run_id: int):
    base_seed = 20251010
    set_seed(base_seed + int(run_id))

    def _natkey(s: str):
        parts = re.split(r'(\d+)', s)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"[MIT] 未在 {DATA_DIR} 下找到任何数据目录")

    subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    subdirs = sorted(subdirs, key=_natkey)

    grouped = {}
    for sub in subdirs:
        subdir = os.path.join(DATA_DIR, sub)
        files = [p for p in glob.glob(os.path.join(subdir, "*.csv"))
                 if "discharge" in os.path.basename(p).lower()]
        files = sorted(files, key=lambda p: _natkey(os.path.basename(p)))
        if files:
            grouped[sub] = files

    if not grouped:
        raise FileNotFoundError(f"[MIT] 未在 {DATA_DIR} 的各子文件夹中找到任何 *discharge*.csv")

    files_tr, files_va, files_te = [], [], []
    for sub, files in grouped.items():
        n = len(files)
        n_tr = int(n * 0.7)
        n_va = int(n * 0.1)
        n_te = max(1, n - n_tr - n_va)
        if n_va == 0 and n >= 3:
            n_va, n_te = 1, max(1, n - n_tr - 1)
        files_tr += files[:n_tr]
        files_va += files[n_tr:n_tr + n_va]
        files_te += files[n_tr + n_va:]

    print(f"[MIT Split by folder][run={run_id}] "
          f"train={len(files_tr)}  val={len(files_va)}  test={len(files_te)} "
          f"(folders={len(grouped)})")

    net_name = str(NETWORK).lower()
    ckpt_dir = os.path.join(".", "checkpoints", "mit", net_name, f"run_{run_id}")
    save_path = os.path.join(ckpt_dir, "best.pt")

    y_true, y_pred, times, cids = train_on_mit(
        files_tr, files_va, files_te,
        device=DEVICE,
        save_best_path=save_path,
        ckpt_dir=ckpt_dir,
        resume=True,
        keep_every_epoch=True
    )

    rmse, mae = compute_metrics(y_true, y_pred)
    mape = compute_mape(y_true, y_pred)
    print(f"[Result][MIT][run={run_id}] MAE={mae:.6f}  MAPE={mape:.6f}  RMSE={rmse:.6f}")

    return float(mae), float(mape), float(rmse)

def _nasa_experiment(run_id: int):
    table_rmses = []
    table_maes  = []

    base_seed = 20251010
    set_seed(base_seed + int(run_id))

    per_file_dir = _results_for_nasa()

    for fname in CSV_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        print(f"\n[Data] {fpath}")
        df = load_one_csv(fpath)

        print("[Stage] Train + Test")
        net_name = str(NETWORK).lower()
        ckpt_dir = os.path.join(".", "checkpoints", "nasa", net_name, os.path.splitext(fname)[0])
        best_path = os.path.join(ckpt_dir, "best.pt")
        y_true, y_pred, times, cids = train_on_nasa(
            df,
            device=DEVICE,
            save_best_path=best_path
        )

        rmse, mae = compute_metrics(y_true, y_pred)
        print(f"[Result][run={run_id}] {fname}  RMSE={rmse:.6f}  MAE={mae:.6f}")
        table_rmses.append(rmse)
        table_maes.append(mae)

        _write_per_file(per_file_dir, fname, run_id=int(run_id),
                               rmse=float(rmse), mae=float(mae))

    rmse_avg = float(np.nanmean(table_rmses)) if table_rmses else float("nan")
    mae_avg  = float(np.nanmean(table_maes)) if table_maes else float("nan")
    n_tables = len(CSV_FILES)
    print(f"\n[Summary][run={run_id}] {n_tables} tables avg  RMSE={rmse_avg:.6f}  MAE={mae_avg:.6f}")
    return rmse_avg, mae_avg

def _results_for_calce() -> str:
    net_name = str(NETWORK).lower()
    root = os.path.join(".", "results", "CALCE_folders", net_name)
    os.makedirs(root, exist_ok=True)
    return root

def _write_calce_result(root_dir: str, folder_name: str, run_id: int, rmse: float, mae: float):
    xlsx_path = os.path.join(root_dir, f"{folder_name}.xlsx")
    cols = ["run", "RMSE", "MAE"]
    try:
        df = pd.read_excel(xlsx_path)
        for c in cols:
            if c not in df.columns:
                df[c] = None
    except Exception:
        df = pd.DataFrame(columns=cols)
    df.loc[len(df)] = {"run": int(run_id), "RMSE": float(rmse), "MAE": float(mae)}
    df.to_excel(xlsx_path, index=False)


def _calce_experiment(run_id: int):
    base_seed = 20251010
    set_seed(base_seed + int(run_id))

    def _natkey(s: str):
        parts = re.split(r'(\d+)', s)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    subdirs = sorted(subdirs, key=_natkey)
    grouped = {}
    for sub in subdirs:
        subdir = os.path.join(DATA_DIR, sub)
        files = [p for p in glob.glob(os.path.join(subdir, "*.csv"))]
        files = sorted(files, key=lambda p: _natkey(os.path.basename(p)))
        if files:
            grouped[sub] = files
    if not grouped:
        raise FileNotFoundError(f"[CALCE] 未在 {DATA_DIR} 的各子文件夹中找到任何 *.csv")

    per_folder_dir = _results_for_calce()

    all_rmses, all_maes = [], []
    for folder, files in grouped.items():
        n = len(files)
        n_tr = int(n * 0.7)
        n_va = int(n * 0.1)
        n_te = max(1, n - n_tr - n_va)
        if n_va == 0 and n >= 3:
            n_va, n_te = 1, max(1, n - n_tr - 1)
        files_tr = files[:n_tr]
        files_va = files[n_tr:n_tr + n_va]
        files_te = files[n_tr + n_va:]

        print(f"\n[CALCE][{folder}] split: train={len(files_tr)} val={len(files_va)} test={len(files_te)} (total={n})")
        net_name = str(NETWORK).lower()
        ckpt_dir = os.path.join(".", "checkpoints", "calce", net_name, f"run_{run_id}", folder)
        best_path = os.path.join(ckpt_dir, "best.pt")
        y_true, y_pred, times, cids = train_on_calce(
            files_tr, files_va, files_te,
            device = DEVICE,
            save_best_path = best_path,
            ckpt_dir = ckpt_dir,
            resume = True,
            keep_every_epoch = True,
        )
        rmse, mae = compute_metrics(y_true, y_pred)
        print(f"[Result][CALCE][run={run_id}][{folder}] RMSE={rmse:.6f} MAE={mae:.6f}")

        all_rmses.append(rmse); all_maes.append(mae)
        _write_calce_result(per_folder_dir, folder, run_id=int(run_id), rmse=float(rmse), mae=float(mae))

    rmse_avg = float(np.nanmean(all_rmses)) if all_rmses else float("nan")
    mae_avg  = float(np.nanmean(all_maes))  if all_maes  else float("nan")
    print(f"\n[Summary][CALCE][run={run_id}] 4 folders avg RMSE={rmse_avg:.6f} MAE={mae_avg:.6f}")
    return rmse_avg, mae_avg


def main():
    print(f"[Device] {DEVICE}")

    results_xlsx = _results_xlsx_for_network()
    start_run = _get_next_run(results_xlsx)
    if start_run > 10:
        print(f"[Info] {results_xlsx} 已经包含 10 次实验")
        return

    if DATASET_NAME == "MIT":
        for run_id in range(start_run, RUNS + 1):
            print(f"\n========== [Experiment run {run_id}/10] ==========")
            mae_avg, mape_avg, rmse_avg = _mit_experiment(run_id)
            _write_total_result_mit(results_xlsx, run_id, mae_avg, mape_avg, rmse_avg)
            print(f"[Export] Appended run={run_id} to {results_xlsx}")
        return

    elif DATASET_NAME == "NASA":
        for run_id in range(start_run, RUNS + 1):
            print(f"\n========== [Experiment run {run_id}/10] ==========")
            rmse_avg, mae_avg = _nasa_experiment(run_id)
            _write_total_result(results_xlsx, run_id, rmse_avg, mae_avg)
            print(f"[Export] Appended run={run_id} to {results_xlsx}")

    elif DATASET_NAME.upper() == "CALCE":
        for run_id in range(1, RUNS + 1):
            print(f"\n========== [Experiment run {run_id}/10] ==========")
            rmse_avg, mae_avg = _calce_experiment(run_id)
            _write_total_result(results_xlsx, run_id, rmse_avg, mae_avg)
            print(f"[Export] Appended run={run_id} to {results_xlsx}")
        exit(0)


if __name__ == "__main__":
    if WHOLE_ENABLE:
        arch  = str(WHOLE_ARCH).lower()
        group = str(UDDS_GROUP).upper()

        xlsx_path = _whole_results(arch, group)
        start = _get_next_run(xlsx_path)

        loaders = build_udds_dataloaders(
            UDDS_DIR, WINDOW_UDDS, STRIDE_UDDS,
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
        )

        for run_id in range(start, RUNS + 1):
            set_seed(int(time.time()) % 1000000)
            print(f"\n========== [Experiment run {run_id}/10] ==========")
            ckpt_root = "checkpoints_whole_distill" if DISTILL_ENABLE else "checkpoints_whole"
            ckpt_dir  = os.path.join(".", ckpt_root, arch, group, f"run_{run_id}")
            best_path = os.path.join(ckpt_dir, "best.pt")

            print(f"[Main][WHOLE] arch={arch} group={group} run={run_id}")
            y_true, y_pred, _, _ = train_whole_on_udds(
                device=DEVICE,
                save_best_path=best_path,
                ckpt_dir=ckpt_dir,
                resume=True,
                keep_every_epoch=True,
                loaders=loaders
            )

            rmse, mae = compute_metrics(y_true, y_pred)
            print(f"[Result][WHOLE][arch={arch}][group={group}][run={run_id}] RMSE={rmse:.6f} MAE={mae:.6f}")
            _write_total_result(xlsx_path, run_id, rmse, mae)
    else:
        main()