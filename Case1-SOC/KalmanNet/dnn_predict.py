import os
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEVICE
from model import CNNGRUAttention

# 滑窗长度
WINDOW_SIZE = 10


def main():
    excel_path = "battery_data.xlsx"
    out_dir = "figures_cy"
    os.makedirs(out_dir, exist_ok=True)

    xls_all = pd.ExcelFile(excel_path)
    all_sheets = xls_all.sheet_names
    if "Capacity" not in all_sheets:
        raise RuntimeError("找不到 Capacity 表。")
    capacity_df = pd.read_excel(excel_path, sheet_name="Capacity")
    if not {"Battery", "Cycle", "Capacity"}.issubset(capacity_df.columns):
        raise RuntimeError("Capacity 表必须包含 Battery, Cycle, Capacity 三列。")

    sheet2df = {}
    for sheet in all_sheets:
        if sheet.startswith("charge_") or sheet.startswith("discharge_"):
            sheet2df[sheet] = pd.read_excel(excel_path, sheet_name=sheet)

    v_min, v_max = float("inf"), float("-inf")
    i_min, i_max = float("inf"), float("-inf")
    for df_sheet in sheet2df.values():
        if not {"Voltage_measured","Current_measured","Time"}.issubset(df_sheet.columns):
            raise RuntimeError("Sheet 缺少必要列。")
        v = df_sheet["Voltage_measured"].values.astype(np.float32)
        i = df_sheet["Current_measured"].values.astype(np.float32)
        v_min = min(v_min, v.min())
        v_max = max(v_max, v.max())
        i_min = min(i_min, i.min())
        i_max = max(i_max, i.max())
    i_abs_max = max(abs(i_min), abs(i_max))
    print(f"[PRED FAST] 全局归一化参数: v_min={v_min:.6f}, v_max={v_max:.6f}, i_abs_max={i_abs_max:.6f}")

    device = torch.device(DEVICE)
    model = CNNGRUAttention(
        in_feats=2,
        cnn_channels=32,
        kernel_size=3,
        pool_size=1,
        gru_hidden=64,
        out_feats=1
    ).to(device)
    ckpt = "checkpoints/cnn_gru_attention_soc.pth"
    if not os.path.isfile(ckpt):
        raise RuntimeError(f"找不到模型文件 {ckpt}，请先训练并保存模型。")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    for sheet, df_sheet in sheet2df.items():
        print(f"\n>>> 处理 sheet: {sheet} <<<")
        is_charge = sheet.startswith("charge_")
        feats_list = []
        time_list = []
        true_list = []
        cycle_slices = []
        start = 0

        for cycle_id, df_cycle in df_sheet.groupby("Cycle"):
            v_arr = df_cycle["Voltage_measured"].values.astype(np.float32)
            i_arr = df_cycle["Current_measured"].values.astype(np.float32)
            t_arr = df_cycle["Time"].values.astype(np.float32)
            T = len(t_arr)
            if T < WINDOW_SIZE:
                print(f"  cycle {cycle_id} 数据长度<{WINDOW_SIZE}，跳过")
                continue

            bid = sheet.split("_")[1]
            cap_row = capacity_df[(capacity_df.Battery==bid)&(capacity_df.Cycle==cycle_id)]
            if cap_row.empty:
                print(f"  WARN: 缺少容量 {bid} cycle={cycle_id}, 跳过")
                continue
            cap = float(cap_row.Capacity.iloc[0])

            soc_true = np.zeros(T, dtype=np.float32)
            q_cum = np.zeros(T, dtype=np.float32)
            soc_true[0] = 0.0 if is_charge else 1.0
            cum = 0.0
            for k in range(1, T):
                dt = max(t_arr[k]-t_arr[k-1],0.0)
                delta = i_arr[k] if is_charge else abs(i_arr[k])
                cum += delta*(dt/3600.0)
                ratio = np.clip(cum/cap,0.0,1.0)
                q_cum[k] = ratio
                soc_true[k] = ratio if is_charge else (1.0-ratio)

            # 归一化特征
            v_norm = np.clip((v_arr - v_min)/(v_max-v_min),0.0,1.0)

            # 构造滑窗样本
            L = T - WINDOW_SIZE + 1
            for idx in range(L):
                win_v = v_norm[idx:idx+WINDOW_SIZE]
                win_q = q_cum[idx:idx+WINDOW_SIZE]
                feat = np.stack([win_v, win_q], axis=1)  # (W,2)
                feats_list.append(feat)
            # 记录对应的时间和真实 SoC
            time_list.extend(t_arr[WINDOW_SIZE-1:])
            true_list.extend(soc_true[WINDOW_SIZE-1:])
            cycle_slices.append((start, start+L, cycle_id))
            start += L

        if start==0:
            print(f"  sheet {sheet} 没有有效数据，跳过")
            continue

        # 批量预测
        all_feats = np.stack(feats_list).astype(np.float32)  # (N_windows, W,2)
        with torch.no_grad():
            input_tensor = torch.from_numpy(all_feats).to(device)
            preds = model(input_tensor).cpu().numpy()  # (N_windows,)

        for (s,e,cycle_id) in cycle_slices:
            seg_time = time_list[s:e]
            seg_true = true_list[s:e]
            seg_pred = preds[s:e]

            plt.figure(figsize=(6,3.5))
            plt.plot(seg_time, seg_true, '--', label='True SoC')
            plt.plot(seg_time, seg_pred, '-', label='Predicted SoC')
            plt.title(f"{sheet} Cycle {cycle_id}")
            plt.xlabel("Time (s)")
            plt.ylabel("SoC")
            plt.grid(True, linestyle=':')
            plt.legend(loc='lower right')
            plt.tight_layout()
            fn = f"{sheet}_cycle_{cycle_id}.png"
            plt.savefig(os.path.join(out_dir, fn), dpi=120)
            plt.close()
        print(f"  已保存 {len(cycle_slices)} 张图到 {out_dir}/")

    print("\n全部循环完毕，图片都保存在", out_dir)

if __name__ == "__main__":
    main()
