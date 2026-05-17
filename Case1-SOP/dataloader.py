import numpy as np
import pandas as pd
from tqdm import tqdm
import os, re, glob, json, hashlib, time
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    WINDOW_NASA, STRIDE_NASA, WINDOW_MIT, STRIDE_MIT, WINDOW_CALCE, STRIDE_CALCE, BATCH_SIZE,
    NUM_WORKERS, CAPACITY_NASA, CAPACITY_CALCE, CAPACITY_MIT, COULOMB_EFFICIENCY, UDDS_GROUP,
    CAPACITY_CONDITION, DATASET_NAME, WHOLE_ENABLE, DISTILL_ENABLE
)

class MIT(Dataset):
    def __init__(self, df: pd.DataFrame, window: int, stride: int, return_cids: bool = False):
        need = {"cycle", "t", "I", "V", "SOC"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"[MIT] 数据缺少列: {sorted(list(miss))}，需要: {sorted(list(need))}")

        g = df[["cycle", "t", "I", "V", "SOC"]].copy()
        g["cycle"] = pd.to_numeric(g["cycle"], errors="coerce").astype(int)
        g["t_sec"] = pd.to_numeric(g["t"], errors="coerce").astype(float) * 3600.0
        g["t_sec"] = g.groupby("cycle")["t_sec"].transform(lambda s: s - float(s.iloc[0]))
        g["I"] = pd.to_numeric(g["I"], errors="coerce").astype(float)
        g["V"] = pd.to_numeric(g["V"], errors="coerce").astype(float)
        g["SOC"] = pd.to_numeric(g["SOC"], errors="coerce").astype(float)
        g = g.dropna().sort_values(["cycle", "t_sec"]).reset_index(drop=True)

        self.cycle = g["cycle"].to_numpy(np.int64)
        self.t_sec = g["t_sec"].to_numpy(np.float32)
        self.I = g["I"].to_numpy(np.float32)
        self.V = g["V"].to_numpy(np.float32)
        self.SOC = g["SOC"].to_numpy(np.float32)

        self.window = int(window)
        self.stride = int(stride)
        self.return_cids = bool(return_cids)
        self.capacity_ah = float(CAPACITY_MIT)

        self.samples = []
        for cid, grp in g.groupby("cycle", sort=False):
            idxs = grp.index.to_numpy()
            a, b = int(idxs[0]), int(idxs[-1]) + 1
            if b - a < self.window:
                continue
            for s in range(a, b - self.window + 1, self.stride):
                e = s + self.window
                self.samples.append((s, e, int(cid)))

    def __len__(self) -> int:
        return len(self.samples)

    def _constrained_current(self, s: int, e: int) -> float:
        I_win = self.I[s:e].astype(float)
        SOC_win = self.SOC[s:e].astype(float)
        t_win = self.t_sec[s:e].astype(float)

        k = int(np.argmax(np.abs(I_win)))
        I_peak = float(I_win[k])

        dsoc = float(np.min(SOC_win) - SOC_win[0])
        dt = float(max(t_win[-1] - t_win[0], 1e-9))
        I_soc = dsoc * (self.capacity_ah * 3600.0) / (COULOMB_EFFICIENCY * dt)

        return float(max(I_peak, I_soc))

    def __getitem__(self, idx: int):
        s, e, cid = self.samples[idx]
        # x: [V, SOC, I]
        x = torch.from_numpy(np.stack([self.V[s:e], self.SOC[s:e], self.I[s:e]], axis=1).astype(np.float32))
        y = torch.tensor(self._constrained_current(s, e), dtype=torch.float32)
        t_end = torch.tensor(self.t_sec[e - 1], dtype=torch.float32)
        if self.return_cids:
            return x, y, t_end, torch.tensor(cid, dtype=torch.long)
        return x, y, t_end

def load_mit(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"cycle", "t", "I", "V", "SOC"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"[MIT] {os.path.basename(path)} 缺少列: {sorted(list(miss))}")
    return df.sort_values(["cycle", "t"]).reset_index(drop=True)

def _concat_mit(files: list) -> pd.DataFrame:
    if not files:
        return pd.DataFrame(columns=["cycle", "t", "I", "V", "SOC"])
    dfs = [load_mit(p) for p in files]
    return pd.concat(dfs, ignore_index=True)

def build_mit_dataloaders(files_tr: list,
                                     files_va: list,
                                     files_te: list,
                                     window: int = WINDOW_MIT,
                                     stride: int = STRIDE_MIT,
                                     batch_size: int = BATCH_SIZE,
                                     num_workers: int = NUM_WORKERS):
    df_tr = _concat_mit(files_tr)
    df_va = _concat_mit(files_va)
    df_te = _concat_mit(files_te)

    ds_tr = MIT(df_tr, window, stride, return_cids=True)
    ds_va = MIT(df_va, window, stride, return_cids=True)
    ds_te = MIT(df_te, window, stride, return_cids=True)

    ltr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
    lva = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    lte = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return ltr, lva, lte

class NASA(Dataset):
    def __init__(self, df: pd.DataFrame, window: int, stride: int, return_cids: bool = False):
        df = df.sort_values(["Discharge_Cycle", "Time(s)"]).reset_index(drop=True)

        self.t   = df["Time(s)"].to_numpy(np.float32)
        self.v   = df["Voltage(V)"].to_numpy(np.float32)
        self.i   = df["Current(A)"].to_numpy(np.float32)
        self.soc = df["SOC"].to_numpy(np.float32)

        self.window = int(window)
        self.stride = int(stride)
        self.return_cids = bool(return_cids)

        self.samples = []
        for cid, g in df.groupby("Discharge_Cycle", sort=False):
            idxs = g.index.to_numpy()
            start = int(idxs[0]); end = int(idxs[-1]) + 1
            if end - start < self.window:
                continue
            for s in range(start, end - self.window + 1, self.stride):
                e = s + self.window
                self.samples.append((s, e, int(cid)))

    def __len__(self):
        return len(self.samples)

    def _constrained_current(self, s: int, e: int) -> float:
        curr_win = self.i[s:e].astype(float)
        soc_win  = self.soc[s:e].astype(float)
        t_win    = self.t[s:e].astype(float)
        idx = int(np.argmax(np.abs(curr_win)))
        I_peak = float(curr_win[idx])
        dsoc = float(np.min(soc_win) - soc_win[0])
        dt   = float(max(t_win[-1] - t_win[0], 1e-9))
        I_soc = dsoc * (CAPACITY_NASA * 3600.0) / (COULOMB_EFFICIENCY * dt)
        return float(max(I_peak, I_soc))

    def __getitem__(self, idx: int):
        s, e, cid = self.samples[idx]
        x = torch.from_numpy(np.stack([self.v[s:e], self.soc[s:e], self.i[s:e]], axis=1).astype(np.float32))
        y = torch.tensor(self._constrained_current(s, e), dtype=torch.float32)
        t_end = torch.tensor(self.t[e - 1], dtype=torch.float32)
        if self.return_cids:
            return x, y, t_end, torch.tensor(cid, dtype=torch.long)
        return x, y, t_end

def build_nasa_dataloaders(df: pd.DataFrame,
                              window: int = WINDOW_NASA,
                              stride: int = STRIDE_NASA,
                              batch_size: int = BATCH_SIZE,
                              num_workers: int = NUM_WORKERS):
    df = df.sort_values(["Discharge_Cycle", "Time(s)"]).reset_index(drop=True)
    sizes = df.groupby("Discharge_Cycle").size()
    valid_cycles = [c for c, L in sizes.items() if L >= window]
    nC = len(valid_cycles)
    c_tr = valid_cycles[: int(0.7 * nC)]
    c_va = valid_cycles[int(0.7 * nC): int(0.8 * nC)]
    c_te = valid_cycles[int(0.8 * nC):]
    df_tr = df[df["Discharge_Cycle"].isin(c_tr)].reset_index(drop=True)
    df_va = df[df["Discharge_Cycle"].isin(c_va)].reset_index(drop=True)
    df_te = df[df["Discharge_Cycle"].isin(c_te)].reset_index(drop=True)

    ds_tr = NASA(df_tr, window, stride, return_cids=True)
    ds_va = NASA(df_va, window, stride, return_cids=True)
    ds_te = NASA(df_te, window, stride, return_cids=True)

    ltr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
    lva = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    lte = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return ltr, lva, lte

def load_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values(["Discharge_Cycle", "Time(s)"]).reset_index(drop=True)

def load_calce(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"time_s", "Voltage(V)", "Current(A)", "SOC(%)"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"[CALCE] {os.path.basename(path)} 缺少列: {sorted(list(miss))}")
    g = pd.DataFrame({
        "t_sec": pd.to_numeric(df["time_s"], errors="coerce").astype(float),
        "V":     pd.to_numeric(df["Voltage(V)"], errors="coerce").astype(float),
        "I":     pd.to_numeric(df["Current(A)"], errors="coerce").astype(float),
        "SOC":   pd.to_numeric(df["SOC(%)"], errors="coerce").astype(float) / 100.0,
    }).dropna().reset_index(drop=True)
    if len(g) > 0:
        g["t_sec"] = g["t_sec"] - float(g["t_sec"].iloc[0])
    return g[["t_sec", "V", "I", "SOC"]]

class CALCE(Dataset):
    def __init__(self, df: pd.DataFrame, window: int, stride: int, return_cids: bool = False):
        need = {"cycle", "t_sec", "I", "V", "SOC"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"[CALCE] 数据缺少列: {sorted(list(miss))}，需要: {sorted(list(need))}")

        g = df[["cycle", "t_sec", "I", "V", "SOC"]].copy()
        g["cycle"] = pd.to_numeric(g["cycle"], errors="coerce").astype(int)
        g["t_sec"] = pd.to_numeric(g["t_sec"], errors="coerce").astype(float)
        g["I"] = pd.to_numeric(g["I"], errors="coerce").astype(float)
        g["V"] = pd.to_numeric(g["V"], errors="coerce").astype(float)
        g["SOC"] = pd.to_numeric(g["SOC"], errors="coerce").astype(float)
        g = g.dropna().sort_values(["cycle", "t_sec"]).reset_index(drop=True)

        self.cycle = g["cycle"].to_numpy(np.int64)
        self.t_sec = g["t_sec"].to_numpy(np.float32)
        self.I     = g["I"].to_numpy(np.float32)
        self.V     = g["V"].to_numpy(np.float32)
        self.SOC   = g["SOC"].to_numpy(np.float32)

        self.window = int(window)
        self.stride = int(stride)
        self.return_cids = bool(return_cids)
        self.capacity_ah = float(CAPACITY_CALCE)

        self.samples = []
        for cid, grp in g.groupby("cycle", sort=False):
            idxs = grp.index.to_numpy()
            a, b = int(idxs[0]), int(idxs[-1]) + 1
            if b - a < self.window:
                continue
            for s in range(a, b - self.window + 1, self.stride):
                e = s + self.window
                self.samples.append((s, e, int(cid)))

    def __len__(self) -> int:
        return len(self.samples)

    def _constrained_current(self, s: int, e: int) -> float:
        I_win   = self.I[s:e].astype(float)
        SOC_win = self.SOC[s:e].astype(float)
        t_win   = self.t_sec[s:e].astype(float)

        k = int(np.argmax(np.abs(I_win)))
        I_peak = float(I_win[k])

        dsoc = float(np.min(SOC_win) - SOC_win[0])
        dt   = float(max(t_win[-1] - t_win[0], 1e-9))
        I_soc = dsoc * (self.capacity_ah * 3600.0) / (COULOMB_EFFICIENCY * dt)

        return float(max(I_peak, I_soc))

    def __getitem__(self, idx: int):
        s, e, cid = self.samples[idx]

        is_calce_voltage = (str(DATASET_NAME).upper() == "CALCE") and (not WHOLE_ENABLE) and (not DISTILL_ENABLE)

        if is_calce_voltage:
            x = torch.from_numpy(np.stack([self.SOC[s:e], self.I[s:e]], axis=1).astype(np.float32))
            y = torch.tensor(float(self.V[e - 1]), dtype=torch.float32)
        else:
            x = torch.from_numpy(np.stack([self.V[s:e], self.SOC[s:e], self.I[s:e]], axis=1).astype(np.float32))
            y = torch.tensor(self._constrained_current(s, e), dtype=torch.float32)

        t_end = torch.tensor(self.t_sec[e - 1], dtype=torch.float32)

        if self.return_cids:
            return x, y, t_end, torch.tensor(cid, dtype=torch.long)
        return x, y, t_end

def _concat_calce(files: list) -> pd.DataFrame:
    frames = []
    for cid, p in enumerate(files):
        g = load_calce(p)
        if len(g) == 0:
            continue
        g = g.copy()
        g.insert(0, "cycle", cid)
        frames.append(g[["cycle", "t_sec", "I", "V", "SOC"]])
    if not frames:
        return pd.DataFrame(columns=["cycle","t_sec","I","V","SOC"])
    return pd.concat(frames, ignore_index=True)

def build_calce_dataloaders(files_tr: list,
                                       files_va: list,
                                       files_te: list,
                                       window: int = WINDOW_CALCE,
                                       stride: int = STRIDE_CALCE,
                                       batch_size: int = BATCH_SIZE,
                                       num_workers: int = NUM_WORKERS):
    df_tr = _concat_calce(files_tr)
    df_va = _concat_calce(files_va)
    df_te = _concat_calce(files_te)

    ds_tr = CALCE(df_tr, window, stride, return_cids=True)
    ds_va = CALCE(df_va, window, stride, return_cids=True)
    ds_te = CALCE(df_te, window, stride, return_cids=True)

    ltr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
    lva = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    lte = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return ltr, lva, lte

class UDDS(Dataset):
    def __init__(self, df: pd.DataFrame, window: int, stride: int, return_cids: bool = False):
        g = df.copy().sort_values(["cycle", "sub_cycle", "Step_Time(s)"]).reset_index(drop=True)

        self.cycle   = g["cycle"].to_numpy(np.int64)
        self.t       = g["Step_Time(s)"].to_numpy(np.float32)
        self.volt    = g["Voltage(V)"].to_numpy(np.float32)
        self.curr    = g["Current(A)"].to_numpy(np.float32)
        self.soc     = g["SOC"].to_numpy(np.float32)
        self.speed   = g["Speed(m/s)"].to_numpy(np.float32)
        self.acc     = g["Acceleration(m/s^2)"].to_numpy(np.float32)
        self.mileage = g["Mileage(m)"].to_numpy(np.float32)

        self.window = int(window)
        self.stride = int(stride)
        self.return_cids = bool(return_cids)

        self.cap_ah = float(CAPACITY_CONDITION)
        self.ceff   = float(COULOMB_EFFICIENCY)

        self.samples = []
        uniq_cycles = g[["cycle", "sub_cycle"]].drop_duplicates().to_numpy()
        for cid, sid in tqdm(uniq_cycles, desc="[UDDS] Index windows", unit="cycle"):
            grp = g[(g["cycle"] == cid) & (g["sub_cycle"] == sid)]
            idx = grp.index.to_numpy()
            a, b = int(idx[0]), int(idx[-1]) + 1
            if b - a < self.window:
                continue
            for s in range(a, b - self.window + 1, self.stride):
                e = s + self.window
                self.samples.append((s, e, int(cid)))

    def __len__(self):
        return len(self.samples)

    def _constrained_current(self, s, e) -> float:
        Iw   = self.curr[s:e].astype(float)
        SOCw = self.soc[s:e].astype(float)
        tw   = self.t[s:e].astype(float)
        I_peak = float(Iw[np.argmax(np.abs(Iw))])
        dt = float(tw[-1] - tw[0]) if e - s > 1 else 1.0
        dsoc = float(np.min(SOCw) - SOCw[0])
        I_soc = dsoc * (self.cap_ah * 3600.0) / max(self.ceff * dt, 1e-9)
        return float(max(I_peak, I_soc))

    def __getitem__(self, idx: int):
        s, e, cid = self.samples[idx]
        xA = torch.from_numpy(np.stack([self.volt[s:e], self.soc[s:e], self.curr[s:e]], axis=1))
        xB = torch.from_numpy(np.stack([self.speed[s:e], self.acc[s:e], self.mileage[s:e]], axis=1))
        y  = torch.tensor(self._constrained_current(s, e), dtype=torch.float32)
        t_end = torch.tensor(self.t[e-1], dtype=torch.float32)
        if self.return_cids:
            return xA, xB, y, t_end, torch.tensor(cid, dtype=torch.long)
        return xA, xB, y, t_end

def build_udds_dataloaders(root: str, window: int, stride: int,
                                    batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):

    group = str(UDDS_GROUP).upper()
    root = os.path.abspath(root)

    subdirs = [d for d in os.listdir(root) if d.lower().startswith("cycling_")]
    def cyc_num(name):
        m = re.search(r'(\d+)$', name)
        return int(m.group(1)) if m else None
    cycles_present = sorted([cyc_num(d) for d in subdirs if cyc_num(d) is not None])

    val_cycles = [1, 2]
    test_map = {
        "A": [3],
        "B": [4],
        "C": [8, 9, 10, 11],
        "D": [12, 13, 14],
    }
    test_cycles = test_map.get(group, [3])

    val = [c for c in val_cycles if c in cycles_present]
    test = [c for c in test_cycles if c in cycles_present]
    train = [c for c in cycles_present if c not in set(val) | set(test)]

    print(f"[UDDS] group={group} | present={cycles_present}")
    print(f"[UDDS] val={val} | test={test} | train={train}")

    usecols = [
        "Step_Time(s)", "Voltage(V)", "Current(A)", "SOC",
        "Speed(m/s)", "Acceleration(m/s^2)", "Mileage(m)"
    ]

    def files_of(cycle_list):
        all_files = []
        for c in cycle_list:
            folder = os.path.join(root, f"Cycling_{c}")
            xs = glob.glob(os.path.join(folder, "*.xlsx"))
            xs.sort(key=lambda p: [int(s) if s.isdigit() else s.lower()
                                   for s in re.split(r'(\d+)', os.path.splitext(os.path.basename(p))[0])])
            all_files.extend([(c, p) for p in xs])
        return all_files

    def _sub_cycle_from_path(p):
        name = os.path.splitext(os.path.basename(p))[0]
        nums = re.findall(r'(\d+)', name)
        if not nums:
            return 0
        return int(nums[-1])

    train_files = files_of(train)
    val_files   = files_of(val)
    test_files  = files_of(test)

    print(f"[UDDS] files: train={len(train_files)} | val={len(val_files)} | test={len(test_files)}")

    def read_concat(tag, pairs):
        frames = []
        with tqdm(total=len(pairs), desc=f"[UDDS] Reading {tag}", unit="file") as pbar:
            for c, p in pairs:
                df = pd.read_excel(p, usecols=usecols, engine="openpyxl")
                df = df.copy()
                df.insert(0, "cycle", c)
                sub_c = _sub_cycle_from_path(p)
                df.insert(1, "sub_cycle", sub_c)
                frames.append(df)
                pbar.update(1)
        if not frames:
            return pd.DataFrame(columns=["cycle", "sub_cycle"] + usecols)
        return pd.concat(frames, ignore_index=True)

    def _sig_of(pairs):
        h = hashlib.sha1()
        for c, p in sorted(pairs, key=lambda x: (x[0], x[1])):
            try:
                st = os.stat(p)
                line = f"{c}|{p}|{int(st.st_mtime)}|{st.st_size}\n"
            except FileNotFoundError:
                line = f"{c}|{p}|MISSING|0\n"
            h.update(line.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    cache_dir = os.path.join(root, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    sig_all = _sig_of(train_files + val_files + test_files)

    cache_pkl = os.path.join(cache_dir, f"udds_{group}.pkl")
    cache_meta = os.path.join(cache_dir, f"udds_{group}.json")

    df_tr = df_va = df_te = None
    if os.path.isfile(cache_pkl) and os.path.isfile(cache_meta):
        try:
            print(f"[UDDS][cache] FORCE hit ({group}) -> {cache_pkl}")
            obj = pd.read_pickle(cache_pkl)
            df_tr, df_va, df_te = obj.get("df_tr"), obj.get("df_va"), obj.get("df_te")
        except Exception as e:
            print(f"[UDDS][cache] load failed: {e}")

    if df_tr is None or df_va is None or df_te is None:
        df_tr = read_concat("Train", train_files)
        df_va = read_concat("Valid", val_files)
        df_te = read_concat("Test", test_files)
        try:
            pd.to_pickle({"df_tr": df_tr, "df_va": df_va, "df_te": df_te}, cache_pkl)
            json.dump({
                "split": group,
                "signature": sig_all,
                "train_files": [p for _, p in train_files],
                "val_files": [p for _, p in val_files],
                "test_files": [p for _, p in test_files],
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, open(cache_meta, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"[UDDS][cache] saved ({group}) -> {cache_pkl}")
        except Exception as e:
            print(f"[UDDS][cache] save failed: {e}")

    ds_tr = UDDS(df_tr, window, stride, return_cids=True)
    ds_va = UDDS(df_va, window, stride, return_cids=True)
    ds_te = UDDS(df_te, window, stride, return_cids=True)

    ltr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
    lva = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    lte = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return ltr, lva, lte