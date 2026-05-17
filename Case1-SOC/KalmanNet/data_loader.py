import pandas as pd
import numpy as np
from config import DATA_PATH

def load_all_trajectories():
    xls = pd.ExcelFile(DATA_PATH, engine='openpyxl')
    if 'Capacity' not in xls.sheet_names:
        raise ValueError("Excel 中缺少名为 'Capacity' 的 sheet")
    df_cap = xls.parse('Capacity')
    for col in ['Battery', 'Cycle', 'Capacity']:
        if col not in df_cap.columns:
            raise KeyError(f"'Capacity' 表缺少列 '{col}'")
    cap_map = {}
    for _, row in df_cap.iterrows():
        b = row['Battery']
        c = row['Cycle']
        cap = row['Capacity']
        cap_map[(b, c)] = cap

    sheet_names = [s for s in xls.sheet_names if s != 'Capacity']
    trajectories = []
    for name in sheet_names:
        df = xls.parse(name)
        for col in ['Battery', 'Cycle', 'Time']:
            if col not in df.columns:
                raise KeyError(f"Sheet '{name}' 缺少列 '{col}'")

        battery_id = df.loc[0, 'Battery']
        cycle_idx = df.loc[0, 'Cycle']
        if (battery_id, cycle_idx) not in cap_map:
            raise KeyError(f"Capacity 表中找不到 Battery={battery_id}, Cycle={cycle_idx} 的记录")
        capacity_ah = cap_map[(battery_id, cycle_idx)]

        is_charge = name.lower().startswith('charge')
        if is_charge:
            voltage_col = 'Voltage_charge'
            current_col = 'Current_charge'
        else:
            voltage_col = 'Voltage_load'
            current_col = 'Current_load'

        if voltage_col not in df.columns or current_col not in df.columns:
            raise KeyError(f"Sheet '{name}' 缺少 '{voltage_col}' 或 '{current_col}' 列")

        time = df['Time'].values.astype(float)        # (T,)
        voltage = df[voltage_col].values.astype(float)  # (T,)
        current_raw = df[current_col].values.astype(float)  # (T,)
        current = np.abs(current_raw)

        T = len(time)
        soc = np.zeros(T, dtype=float)
        cum_ah = 0.0
        if is_charge:
            soc[0] = 0.0
            for t in range(1, T):
                dt = time[t] - time[t-1]
                if dt < 0:
                    dt = 0.0
                cum_ah += current[t] * (dt / 3600.0)
                soc[t] = min(cum_ah / capacity_ah, 1.0)
        else:
            soc[0] = 1.0
            for t in range(1, T):
                dt = time[t] - time[t-1]
                if dt < 0:
                    dt = 0.0
                cum_ah += current[t] * (dt / 3600.0)
                soc[t] = max(1.0 - cum_ah / capacity_ah, 0.0)

        trajectories.append({
            'id': name,
            'time': time,         # (T,)
            'voltage': voltage,   # (T,)
            'current': current,   # (T,)
            'soc': soc,            # (T,)
            'capacity': capacity_ah
        })

    return trajectories

if __name__ == '__main__':
    trajs = load_all_trajectories()
    print(f"共加载 {len(trajs)} 条轨迹：")
    for tr in trajs:
        print(f"  - {tr['id']}，长度 = {len(tr['time'])}，SoC[0:3] = {tr['soc'][0:3]}")
