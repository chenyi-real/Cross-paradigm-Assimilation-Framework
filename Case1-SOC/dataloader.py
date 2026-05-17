import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# from config import WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS
# from config_mit import WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS
COULOMB_EFFICIENCY = 1
CAPACITY_CONDITION = 4.93
import re
from datetime import datetime, timedelta

import hashlib

# def _detect_cycle_col(df: pd.DataFrame):
#     return CYCLE_COL_NAME if CYCLE_COL_NAME in df.columns else None

class NasaSeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int, stride: int, return_cids: bool = False):
        super().__init__()

        self.window = int(window)
        self.stride = int(stride)
        self.return_cids = bool(return_cids)

        cycle_col = "Discharge_Cycle"

        df = df.sort_values(["Discharge_Cycle", "Time(s)"]).reset_index(drop=True)

        self.x_raw = df[["Time(s)", "Voltage(V)", "Current(A)"]].to_numpy(np.float32)
        self.y_raw = df["SOC"].to_numpy(np.float32)
        self.t_raw = df["Time(s)"].to_numpy(np.float32)
        mu = self.x_raw.mean(axis=0, keepdims=True)
        sigma = self.x_raw.std(axis=0, keepdims=True).clip(min=1e-6)
        self.x_raw = (self.x_raw - mu) / sigma

        self.starts: List[int] = []
        self.cids: List[int] = []

        if cycle_col is not None:
            for cid, g in df.groupby(cycle_col, sort=False):
                idx = g.index.to_numpy()
                if len(idx) < self.window:
                    continue
                for s_local in range(0, len(idx) - self.window + 1, self.stride):
                    s_global = idx[s_local]
                    self.starts.append(s_global)
                    self.cids.append(int(cid))
        else:
            n = len(df)
            for s in range(0, max(0, n - self.window + 1), self.stride):
                self.starts.append(s)
                self.cids.append(-1)

        self.starts = np.asarray(self.starts, dtype=np.int64)
        self.cids = np.asarray(self.cids, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i: int):
        s = int(self.starts[i]); e = s + self.window
        x = self.x_raw[s:e]
        y = self.y_raw[e - 1]
        x = torch.from_numpy(x)
        y = torch.tensor(y, dtype=torch.float32)
        t = torch.tensor(self.t_raw[e - 1], dtype=torch.float32)
        if self.return_cids:
            return x, y, t, torch.tensor(int(self.cids[i]), dtype=torch.long)
        return x, y, t


class UDDSSeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int, stride: int, return_cids: bool = False, if_demo: bool = False):
        super().__init__()

        self.if_demo = if_demo # demo is True
        df = df.sort_values(['Order_ID', 'Cell_ID', 'Cycle_Number', "Step_Time(s)"]).reset_index(drop=True)
        # df = df.sort_values(['Order_ID', "Step_Time(s)"]).reset_index(drop=True)
        
        # SOP copy
        # self.cycle   = df['Order_ID'].to_numpy(np.int64)
        self.t       = df["Step_Time(s)"].to_numpy(np.float32)
        self.volt    = df["Voltage(V)"].to_numpy(np.float32)
        self.curr    = df["Current(A)"].to_numpy(np.float32)
        self.soc     = df["SOC"].to_numpy(np.float32)
        self.speed   = df["Speed(m/s)"].to_numpy(np.float32)
        self.acc     = df["Acceleration(m/s^2)"].to_numpy(np.float32)
        self.mileage = df["Mileage(m)"].to_numpy(np.float32)


        self.window = int(window)
        self.stride = int(stride)
        self.return_cids = bool(return_cids)

        self.cap_ah = float(CAPACITY_CONDITION)
        self.ceff   = float(COULOMB_EFFICIENCY)

        cycle_col = 'Order_ID'
        df["Step_Time(s)"] = df["Step_Time(s)"]
        self.x_raw = df[["Step_Time(s)", "Voltage(V)", "Current(A)"]].to_numpy(np.float32)
        self.y_raw = df["SOC"].to_numpy(np.float32)
        self.t_raw = df["Step_Time(s)"].to_numpy(np.float32)
        mu = self.x_raw.mean(axis=0, keepdims=True)
        sigma = self.x_raw.std(axis=0, keepdims=True).clip(min=1e-6)
        self.x_raw = (self.x_raw - mu) / sigma

        self.starts: List[int] = []
        self.cids: List[int] = []

        
        for cid, g in df.groupby(cycle_col, sort=False):
            if type(cid) is str:
                cid = int(cid[1:])
            idx = g.index.to_numpy()
            if len(idx) < self.window:
                continue
            for s_local in range(0, len(idx) - self.window + 1, self.stride):
                s_global = idx[s_local]
                self.starts.append(s_global)
                self.cids.append(int(cid))
        

        self.starts = np.asarray(self.starts, dtype=np.int64)
        self.cids = np.asarray(self.cids, dtype=np.int64)


    def __len__(self):
        return len(self.starts)


    def _I_star(self, s, e) -> float:
        Iw   = self.curr[s:e].astype(float)
        SOCw = self.soc[s:e].astype(float)
        tw   = self.t[s:e].astype(float)
        I_peak = float(Iw[np.argmax(np.abs(Iw))])
        dt = float(tw[-1] - tw[0]) if e - s > 1 else 1.0
        dsoc = float(np.min(SOCw) - SOCw[0])
        I_soc = dsoc * (self.cap_ah * 3600.0) / max(self.ceff * dt, 1e-9)
        return float(max(I_peak, I_soc))


    def __getitem__(self, i: int):
        s = int(self.starts[i]); e = s + self.window

        # SOC data
        soc_x = self.x_raw[s:e]
        soc_y = self.y_raw[e - 1]
        soc_x = torch.from_numpy(soc_x)
        soc_y = torch.tensor(soc_y, dtype=torch.float32)
        soc_t = torch.tensor(self.t_raw[e - 1], dtype=torch.float32)

        # SOP data
        # s = s + 1 # 这两句话做了一个偏移
        # e = e + 1 # 
        sop_xA = torch.from_numpy(np.stack([self.volt[s:e], self.soc[s:e], self.curr[s:e]], axis=1))
        sop_xB = torch.from_numpy(np.stack([self.speed[s:e], self.acc[s:e], self.mileage[s:e]], axis=1))
        sop_y  = torch.tensor(self._I_star(s, e), dtype=torch.float32)
        sop_t_end = torch.tensor(self.t[e-1], dtype=torch.float32)

        if self.if_demo:
            # SOC+SOP assenble
            soc_out = (soc_x, soc_y, soc_t)
            sop_out = (sop_xA, sop_xB, sop_y, sop_t_end)
            if self.return_cids:
                return soc_out, sop_out, torch.tensor(int(self.cids[i]), dtype=torch.long)
            return soc_out, sop_out
        else:
            if self.return_cids:
                return soc_x, soc_y, soc_t, torch.tensor(int(self.cids[i]), dtype=torch.long)
            return soc_x, soc_y, soc_t



class MITSeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int, stride: int, return_cids: bool = False):
        super().__init__()

        self.window = int(window)
        self.stride = int(stride)
        self.return_cids = bool(return_cids)

        cycle_col = "cycle"

        df = df.sort_values(['batch_date', "cycle", "t"]).reset_index(drop=True)
        df["t"] = df["t"] * 60.0
        self.x_raw = df[["t", "V", "I"]].to_numpy(np.float32)
        self.y_raw = df["SOC"].to_numpy(np.float32)
        self.t_raw = df["t"].to_numpy(np.float32)
        mu = self.x_raw.mean(axis=0, keepdims=True)
        sigma = self.x_raw.std(axis=0, keepdims=True).clip(min=1e-6)
        self.x_raw = (self.x_raw - mu) / sigma

        self.starts: List[int] = []
        self.cids: List[int] = []


        if cycle_col is not None:
            # 先按 Date 分组
            for date_val, df_date in df.groupby("batch_date", sort=False):
                # 再按 cycle 分组
                for cid, g in df_date.groupby(cycle_col, sort=False):
                    idx = g.index.to_numpy()
                    if len(idx) < self.window:
                        continue
                    for s_local in range(0, len(idx) - self.window + 1, self.stride):
                        s_global = idx[s_local]
                        self.starts.append(s_global)
                        self.cids.append(int(cid))

        else:
            n = len(df)
            for s in range(0, max(0, n - self.window + 1), self.stride):
                self.starts.append(s)
                self.cids.append(-1)

        self.starts = np.asarray(self.starts, dtype=np.int64)
        self.cids = np.asarray(self.cids, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i: int):
        s = int(self.starts[i]); e = s + self.window
        x = self.x_raw[s:e]
        y = self.y_raw[e - 1]
        x = torch.from_numpy(x)
        y = torch.tensor(y, dtype=torch.float32)
        t = torch.tensor(self.t_raw[e - 1], dtype=torch.float32)
        if self.return_cids:
            return x, y, t, torch.tensor(int(self.cids[i]), dtype=torch.long)
        return x, y, t


class CS2SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int, stride: int, return_cids: bool = False):
        super().__init__()

        self.window = int(window)
        self.stride = int(stride)
        self.return_cids = bool(return_cids)

        cycle_col = "cycle"

        df = df.sort_values(['Date', "cycle", "time_s"]).reset_index(drop=True)

        self.x_raw = df[["time_s", "Voltage(V)", "Current(A)"]].to_numpy(np.float32)
        self.y_raw = df["SOC(%)"].to_numpy(np.float32) /100.0
        self.t_raw = df["time_s"].to_numpy(np.float32)
        mu = self.x_raw.mean(axis=0, keepdims=True)
        sigma = self.x_raw.std(axis=0, keepdims=True).clip(min=1e-6)
        self.x_raw = (self.x_raw - mu) / sigma

        self.starts: List[int] = []
        self.cids: List[int] = []

        if cycle_col is not None:
            # 先按 Date 分组
            for date_val, df_date in df.groupby("Date", sort=False):
                # 再按 cycle 分组
                for cid, g in df_date.groupby(cycle_col, sort=False):
                    idx = g.index.to_numpy()
                    if len(idx) < self.window:
                        continue
                    for s_local in range(0, len(idx) - self.window + 1, self.stride):
                        s_global = idx[s_local]
                        self.starts.append(s_global)
                        self.cids.append(int(cid))
        else:
            n = len(df)
            for s in range(0, max(0, n - self.window + 1), self.stride):
                self.starts.append(s)
                self.cids.append(-1)

        self.starts = np.asarray(self.starts, dtype=np.int64)
        self.cids = np.asarray(self.cids, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i: int):
        s = int(self.starts[i]); e = s + self.window
        x = self.x_raw[s:e]
        y = self.y_raw[e - 1]
        x = torch.from_numpy(x)
        y = torch.tensor(y, dtype=torch.float32)
        t = torch.tensor(self.t_raw[e - 1], dtype=torch.float32)
        if self.return_cids:
            return x, y, t, torch.tensor(int(self.cids[i]), dtype=torch.long)
        return x, y, t


def _split_indices(n: int, ratios=(0.7, 0.1, 0.2)):
    assert abs(sum(ratios) - 1.0) < 1e-6
    n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
    idx = np.arange(n)
    i_tr = idx[:n_tr]
    i_va = idx[n_tr:n_tr + n_va]
    i_te = idx[n_tr + n_va:]
    return i_tr, i_va, i_te

def build_dataloaders_from_df(df, window=64, stride=8,
                              batch_size=128, num_workers=0):
    cycle_col = "Discharge_Cycle" if "Discharge_Cycle" in df.columns else None
    if cycle_col is not None:
        df = df.sort_values([cycle_col, "Time(s)"]).reset_index(drop=True)
    else:
        df = df.sort_values(["Time(s)"]).reset_index(drop=True)

    lengths = df.groupby(cycle_col).size() if cycle_col else pd.Series({-1: len(df)})
    valid_cycles = [int(c) for c, L in lengths.items() if L >= window]

    nC = len(valid_cycles)
    c_tr = valid_cycles[: int(0.7*nC)]
    c_va = valid_cycles[int(0.7*nC): int(0.8*nC)]
    c_te = valid_cycles[int(0.8*nC):]

    df_tr = df[df[cycle_col].isin(c_tr)].reset_index(drop=True)
    df_va = df[df[cycle_col].isin(c_va)].reset_index(drop=True)
    df_te = df[df[cycle_col].isin(c_te)].reset_index(drop=True)

    ds_tr = NasaSeqDataset(df_tr, window, stride, return_cids=False)
    ds_va = NasaSeqDataset(df_va, window, stride, return_cids=False)
    ds_te = NasaSeqDataset(df_te, window, stride, return_cids=True)

    ltr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
    lva = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    lte = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return ltr, lva, lte, df_te


def build_dataloaders_from_mit_df(df, window=64, stride=8,
                              batch_size=128, num_workers=8):
    cycle_col = "cycle" if "cycle" in df.columns else None
    if cycle_col is not None:
        df = df.sort_values(['batch_date', cycle_col, "t"]).reset_index(drop=True)
    else:
        df = df.sort_values(['batch_date', "t"]).reset_index(drop=True)

    lengths = df.groupby('batch_date').size() if 'batch_date' else pd.Series({-1: len(df)})
    valid_cycles = [c for c, L in lengths.items() if L >= window]
    nC = len(valid_cycles)
    print(nC)
    c_tr = valid_cycles[: int(0.7*nC)]
    c_va = valid_cycles[int(0.7*nC): int(0.8*nC)]
    c_te = valid_cycles[int(0.8*nC):]

    df_tr = df[df['batch_date'].isin(c_tr)].reset_index(drop=True)
    df_va = df[df['batch_date'].isin(c_va)].reset_index(drop=True)
    df_te = df[df['batch_date'].isin(c_te)].reset_index(drop=True)

    ds_tr = MITSeqDataset(df_tr, window, stride, return_cids=False)
    ds_va = MITSeqDataset(df_va, window, stride, return_cids=False)
    ds_te = MITSeqDataset(df_te, window, stride, return_cids=True)

    ltr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
    lva = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    lte = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return ltr, lva, lte, df_te


def build_dataloaders_from_cs2_df(df, window=64, stride=8,
                              batch_size=128, num_workers=0):
    cycle_col = "cycle" if "cycle" in df.columns else None
    if cycle_col is not None:
        df = df.sort_values(['Date', cycle_col, "time_s"]).reset_index(drop=True)
    else:
        df = df.sort_values(['Date', "time_s"]).reset_index(drop=True)

    lengths = df.groupby(cycle_col).size() if cycle_col else pd.Series({-1: len(df)})
    valid_cycles = [int(c) for c, L in lengths.items() if L >= window]

    nC = len(valid_cycles)
    c_tr = valid_cycles[: int(0.7*nC)]
    c_va = valid_cycles[int(0.7*nC): int(0.8*nC)]
    c_te = valid_cycles[int(0.8*nC):]

    df_tr = df[df[cycle_col].isin(c_tr)].reset_index(drop=True)
    df_va = df[df[cycle_col].isin(c_va)].reset_index(drop=True)
    df_te = df[df[cycle_col].isin(c_te)].reset_index(drop=True)

    ds_tr = CS2SeqDataset(df_tr, window, stride, return_cids=False)
    ds_va = CS2SeqDataset(df_va, window, stride, return_cids=False)
    ds_te = CS2SeqDataset(df_te, window, stride, return_cids=True)

    ltr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
    lva = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    lte = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return ltr, lva, lte, df_te


def build_dataloaders_from_udds_df(train_df, val_df, test_df, window=64, stride=8,
                              batch_size=128, num_workers=8, if_demo=False):
    # cycle_col = "cycle" if "cycle" in df.columns else None
    # if cycle_col is not None:
    #     df = df.sort_values(['batch_date', cycle_col, "t"]).reset_index(drop=True)
    # else:
    #     df = df.sort_values(['batch_date', "t"]).reset_index(drop=True)

    # lengths = df.groupby('batch_date').size() if 'batch_date' else pd.Series({-1: len(df)})
    # valid_cycles = [c for c, L in lengths.items() if L >= window]
    # nC = len(valid_cycles)
    # print(nC)
    # c_tr = valid_cycles[: int(0.7*nC)]
    # c_va = valid_cycles[int(0.7*nC): int(0.8*nC)]
    # c_te = valid_cycles[int(0.8*nC):]


    # df_tr = df[df['batch_date'].isin(c_tr)].reset_index(drop=True)
    # df_va = df[df['batch_date'].isin(c_va)].reset_index(drop=True)
    # df_te = df[df['batch_date'].isin(c_te)].reset_index(drop=True)

    ds_tr = UDDSSeqDataset(train_df, window, stride, return_cids=False, if_demo=if_demo)
    ds_va = UDDSSeqDataset(val_df, window, stride, return_cids=False, if_demo=if_demo)
    ds_te = UDDSSeqDataset(test_df, window, stride, return_cids=True, if_demo=if_demo)

    ltr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
    lva = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    lte = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return ltr, lva, lte


# NASA READ
def load_one_csv(path: str):
    df = pd.read_csv(path)
    cycle_col = "Discharge_Cycle"
    if cycle_col is not None:
        return df.sort_values([cycle_col, "Time(s)"]).reset_index(drop=True)
    return df.sort_values(["Time(s)"]).reset_index(drop=True)

# CS2 READ
def load_all_csv_in_folder(folder_path):
    all_data = []  # 用于存储所有的 CSV 数据
    
    # 获取文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # 按下划线分割文件名，提取日期信息
            parts = filename.split('_')
            if len(parts) >= 5:
                month = parts[2]  # 第四个部分是月份
                day = parts[3]    # 第五个部分是日期
                year = parts[4]  # 文件名中年份在第六个部分，去掉文件扩展名
                
                # 创建标准的日期格式 YYYY-MM-DD
                date = f"20{year}-{month.zfill(2)}-{day.zfill(2)}"
            else:
                continue  # 如果文件名格式不符合预期，跳过此文件
            
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            # 将日期添加为一列
            df['Date'] = date
            # 将当前文件的数据添加到所有数据列表中
            all_data.append(df)
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 排序：按日期、Discharge_Cycle、Time(s)
    combined_df = combined_df.sort_values(by=['Date', 'cycle', 'time_s']).reset_index(drop=True)
    
    return combined_df


def _battery_idx(name: str) -> int:
    """提取文件名中的 battery 编号"""
    m = re.search(r"battery[-_](\d+)", name)
    return int(m.group(1)) if m else 10**9


def load_MIT_csv_data(folder_path):
    """
    从指定文件夹读取所有 MIT csv 文件，
    仅保留列 ["cycle", "t", "I", "V", "SOC"]，
    记录 battery_id，且不同 battery 的 cycle 不进行偏移。
    """
    all_data = []
    useful_cols = ["cycle", "t", "I", "V", "SOC"]

    # 获取文件夹中的所有 CSV 文件
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    files = sorted(files, key=_battery_idx)

    # 获取当前文件夹的日期部分 (如 "2017-05-30")
    base_date = datetime.strptime(folder_path.split('/')[-1], "%Y-%m-%d")

    for idx, file in enumerate(tqdm(files, desc="Processing files", unit="file", leave=False)):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # 仅保留有用列
        df = df[useful_cols]

        # 提取 battery_id
        battery_id = _battery_idx(file)
        df["battery_id"] = battery_id

        # 日期按文件顺序递增
        current_date = base_date + timedelta(days=idx)
        df["batch_date"] = current_date.strftime("%Y-%m-%d")

        all_data.append(df)

    # 合并所有数据
    full_df = pd.concat(all_data, ignore_index=True)

    # 转换日期并排序
    full_df["batch_date"] = pd.to_datetime(full_df["batch_date"])
    full_df = full_df.sort_values(by=["batch_date", "cycle", "t"]).reset_index(drop=True)

    return full_df




def _get_folder_signature(root, folder_paths):
    """
    根据文件名与修改时间生成哈希签名，用于检测文件变化。
    """
    md5 = hashlib.md5()
    for folder_name in folder_paths:
        folder_path = os.path.join(root, folder_name)
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(folder_path, filename)
                md5.update(filename.encode())
                md5.update(str(os.path.getmtime(file_path)).encode())
    return md5.hexdigest()


def load_UDDS_csv_data(root, folder_paths, fname='A', split='train', use_cache=True):
    """
    从多个文件夹读取Excel文件并合并为DataFrame，结果缓存为 .parquet。
    缓存文件命名格式为：combined_cache_{fname}_{split}.parquet / .sig。

    参数:
        root (str): 根目录
        folder_paths (list[str]): 文件夹列表，例如 ['Cycling_1', 'Cycling_2']
        fname (str): 数据标识符（例如 'A', 'B', 'C', 'D'）
        split (str): 数据集划分标识（'train' / 'val' / 'test'）
        use_cache (bool): 是否启用缓存（默认True）

    返回:
        combined_df (pd.DataFrame)
    """
    cache_path = os.path.join(root, f"combined_cache_{fname}_{split}.parquet")
    sig_path   = os.path.join(root, f"combined_sig_{fname}_{split}.txt")

    # ===== Step 1 检测缓存 =====
    current_sig = _get_folder_signature(root, folder_paths)
    if use_cache and os.path.exists(cache_path) and os.path.exists(sig_path):
        with open(sig_path, "r") as f:
            old_sig = f.read().strip()
        if old_sig == current_sig:
            print(f"✅ 直接加载缓存: {os.path.basename(cache_path)}")
            return pd.read_parquet(cache_path)
        else:
            print(f"⚠️ 文件有更新，重新读取数据: {split}")

    # ===== Step 2 读取所有Excel文件 =====
    all_data = []
    for folder_name in folder_paths:
        folder_path = os.path.join(root, folder_name)
        if not os.path.exists(folder_path):
            print(f"⚠️ 文件夹 {folder_name} 不存在，跳过。")
            continue

        file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
        if not file_list:
            print(f"⚠️ 文件夹 {folder_name} 中没有 xlsx 文件。")
            continue

        for filename in tqdm(file_list, desc=f"📂 读取 {split}:{folder_name}"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_excel(file_path)

                # 解析文件名结构: UDDS_C1_W8_001.xlsx
                parts = filename.split('_')
                if len(parts) >= 4:
                    test_type = parts[0]
                    order_id  = parts[1]
                    cell_id   = parts[2]
                    cycle_num = parts[3].split('.')[0]
                else:
                    test_type, order_id, cell_id, cycle_num = [None]*4

                df['Source_Folder'] = folder_name
                df['Source_File']   = filename
                df['Test_Type']     = test_type
                df['Order_ID']      = order_id
                df['Cell_ID']       = cell_id
                df['Cycle_Number']  = cycle_num

                all_data.append(df)
            except Exception as e:
                print(f"❌ 读取失败: {file_path}, 错误：{e}")

    if not all_data:
        raise ValueError(f"{split} 数据集中未找到可读取的Excel文件。")

    # ===== Step 3 合并与排序 =====
    combined_df = pd.concat(all_data, ignore_index=True)
    sort_cols = [c for c in ['Order_ID', 'Cell_ID', 'Cycle_Number', 'Step_Time(s)'] if c in combined_df.columns]
    combined_df = combined_df.sort_values(by=sort_cols).reset_index(drop=True)

    # ===== Step 4 写入缓存 =====
    if use_cache:
        combined_df.to_parquet(cache_path, index=False)
        with open(sig_path, "w") as f:
            f.write(current_sig)
        print(f"💾 缓存已保存: {os.path.basename(cache_path)}")

    return combined_df
