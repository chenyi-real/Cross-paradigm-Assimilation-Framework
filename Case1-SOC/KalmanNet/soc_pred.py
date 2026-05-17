import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config import DEVICE
from model import KalmanNetNet1, CNNGRUAttention
from kalman_filter import KalmanFilterWrapperDNN
from features import compute_kalman_priors, compute_F1_F2_F4
from sklearn.metrics import mean_squared_error

WINDOW_SIZE = 10
C_N = 2.0          # 额定容量 Ah
ETA = 1.0          # 库仑效率
L_SAMP = 100        # 采样点数


def compute_rmse(true_values, predicted_values):
    return np.sqrt(mean_squared_error(true_values, predicted_values))


def compute_soc_and_features_and_dnn_pred(
    df_cycle,
    capacity_ah,
    v_min, v_max, i_abs_max,
    dnn_model,
    is_charge=True
):
    df_cycle = df_cycle.sort_values("Time").reset_index(drop=True)

    time_arr = df_cycle["Time"].values.astype(np.float32)
    v_arr    = df_cycle["Voltage_measured"].values.astype(np.float32)
    i_arr    = df_cycle["Current_measured"].values.astype(np.float32)
    T = len(time_arr)
    if T < WINDOW_SIZE:
        raise RuntimeError(f"数据长度 {T} 小于滑窗 {WINDOW_SIZE}")

    soc_true = np.zeros(T, dtype=np.float32)
    cum = 0.0
    if is_charge:
        soc_true[0] = 0.0
        for t in range(1, T):
            dt = max(time_arr[t] - time_arr[t-1], 0.0)
            cum += abs(i_arr[t]) * (dt/3600.0)
            soc_true[t] = min(cum/capacity_ah, 1.0)
    else:
        soc_true[0] = 1.0
        for t in range(1, T):
            dt = max(time_arr[t] - time_arr[t-1], 0.0)
            cum += abs(i_arr[t]) * (dt/3600.0)
            soc_true[t] = max(1.0 - cum/capacity_ah, 0.0)

    v_norm = ((v_arr - v_min)/(v_max - v_min)).clip(0.0,1.0)
    q_cum = np.zeros(T, dtype=np.float32)
    cum2 = 0.0
    for t in range(1, T):
        dt = max(time_arr[t] - time_arr[t-1], 0.0)
        cum2 += abs(i_arr[t]) * (dt/3600.0)
        q_cum[t] = np.clip(cum2/capacity_ah, 0.0, 1.0)
    feats = np.stack([v_norm, q_cum], axis=1).astype(np.float32)

    W = WINDOW_SIZE
    windows = np.stack([feats[i:i+W] for i in range(T-W+1)], axis=0)
    with torch.no_grad():
        inp = torch.from_numpy(windows).to(DEVICE)
        y_pred_win = dnn_model(inp).cpu().numpy().squeeze(-1)
    soc_pred_dnn = np.empty(T, dtype=np.float32)
    soc_pred_dnn[:W-1] = y_pred_win[0]
    soc_pred_dnn[W-1:] = y_pred_win

    # 计算 KalmanNet 差分特征
    priors = compute_kalman_priors({"soc": soc_pred_dnn})
    F1, F2, F4 = compute_F1_F2_F4({"soc": soc_pred_dnn}, priors)
    X_diff = np.concatenate([F1, F2, F4], axis=1).astype(np.float32)

    # 修正 dt
    raw_dt = np.diff(time_arr, prepend=time_arr[0])
    median_dt = float(np.median(raw_dt[raw_dt > 1e-3])) if np.any(raw_dt > 1e-3) else 1.0
    dt_vals = np.where(raw_dt > 1e-3, raw_dt, median_dt)

    return time_arr, soc_true, X_diff, soc_pred_dnn, v_arr, i_arr, dt_vals


if __name__ == "__main__":
    kalm_ckpt = os.path.join("checkpoints", "final.pth")
    kalman_net = KalmanNetNet1().to(DEVICE)
    kalman_net.load_state_dict(torch.load(kalm_ckpt, map_location=DEVICE))
    kalman_net.eval()
    kf_wrapper = KalmanFilterWrapperDNN(kalman_net).eval()

    dnn_ckpt = os.path.join("checkpoints", "cnn_gru_attention_soc.pth")
    dnn_model = CNNGRUAttention(in_feats=2, cnn_channels=32,
                                kernel_size=3, pool_size=1,
                                gru_hidden=64, out_feats=1).to(DEVICE)
    dnn_model.load_state_dict(torch.load(dnn_ckpt, map_location=DEVICE))
    dnn_model.eval()

    xls = pd.ExcelFile("battery_data.xlsx", engine="openpyxl")
    cap_df = xls.parse("Capacity") if "Capacity" in xls.sheet_names else None
    cap_map = {(r.Battery, int(r.Cycle)): float(r.Capacity) for _, r in cap_df.iterrows()} if cap_df is not None else {}

    v_min, v_max, i_abs_max = 0.236356, 8.393141, 2.026182

    for sheet in xls.sheet_names:
        if sheet == "Capacity":
            continue
        df = xls.parse(sheet)
        if df.empty:
            continue

        battery = df.loc[0, "Battery"]
        is_charge = sheet.lower().startswith("charge")

        rmse_dnn_cycle = []
        rmse_kf_cycle  = []
        rmse_sop_cycle = []

        for cycle in sorted(df.Cycle.unique()):
            # 第33个cycle数据有问题直接跳过
            if cycle == 33:
                continue
            key = (battery, int(cycle))
            if key not in cap_map:
                continue
            df_c = df[df.Cycle == cycle].reset_index(drop=True)

            try:
                t, soc_true, X_diff, soc_pred_dnn, v_arr, i_arr, dt_vals = \
                    compute_soc_and_features_and_dnn_pred(
                        df_c, cap_map[key], v_min, v_max, i_abs_max,
                        dnn_model, is_charge
                    )
            except Exception as e:
                print(f"WARN {sheet} cycle{cycle}: {e}")
                continue

            I_b = torch.from_numpy(i_arr).unsqueeze(0).float().to(DEVICE)
            dt_b = torch.from_numpy(dt_vals).unsqueeze(0).float().to(DEVICE)
            y_b  = torch.from_numpy(soc_pred_dnn).unsqueeze(0).float().to(DEVICE)
            cap_b= torch.tensor([cap_map[key]]).view(1,1).float().to(DEVICE)
            X_b  = torch.from_numpy(X_diff[None]).to(DEVICE)
            with torch.no_grad():
                kf_wrapper.init_soc = y_b[:, :1]
                kf_wrapper.use_dnn_init = True
                est_seq, _ = kf_wrapper(X_b, y_b, I_b, dt_b, cap_b)
                soc_kf = est_seq.squeeze(-1).cpu().numpy().squeeze(0)

            dt_sec = np.mean(dt_vals)
            dt_hr = dt_sec / 3600.0

            # 真实 SOP_true
            I_true = np.empty_like(soc_true, dtype=np.float32)
            for idx in range(len(soc_true)):
                end = min(idx + L_SAMP, len(soc_true))
                window = soc_true[idx:end]
                if is_charge:
                    peak = window.max()
                    delta = peak - soc_true[idx]
                else:
                    valley = window.min()
                    delta = soc_true[idx] - valley
                I_true[idx] = C_N * delta / (ETA * L_SAMP * dt_hr)
            SOP_true = v_arr * I_true

            # 预测 SOP_kf
            I_kf = np.empty_like(soc_kf, dtype=np.float32)
            for idx in range(len(soc_kf)):
                end = min(idx + L_SAMP, len(soc_kf))
                window = soc_kf[idx:end]
                if is_charge:
                    peak = window.max()
                    delta = peak - soc_kf[idx]
                else:
                    valley = window.min()
                    delta = soc_kf[idx] - valley
                I_kf[idx] = C_N * delta / (ETA * L_SAMP * dt_hr)
            SOP_kf = v_arr * I_kf

            start = WINDOW_SIZE - 1
            rmse_dnn_val = compute_rmse(soc_true[start:], soc_pred_dnn[start:])
            rmse_kf_val  = compute_rmse(soc_true[start:], soc_kf[start:])
            rmse_sop_raw = compute_rmse(SOP_true[start:], SOP_kf[start:])
            norm_factor = np.max(np.abs(SOP_true[start:])) if np.any(SOP_true[start:]) else 1.0
            rmse_sop_val = rmse_sop_raw / norm_factor

            rmse_dnn_cycle.append(rmse_dnn_val)
            rmse_kf_cycle.append(rmse_kf_val)
            rmse_sop_cycle.append(rmse_sop_val)


            t_plot = t[start:]
            soc_true_plot = soc_true[start:]
            soc_dnn_plot = soc_pred_dnn[start:]
            soc_kf_plot = soc_kf[start:]
            """
            plt.figure(figsize=(8, 4))
            plt.plot(t_plot, soc_true_plot, '--', label='True SoC')
            plt.plot(t_plot, soc_dnn_plot, ':', label='DNN SoC')
            plt.plot(t_plot, soc_kf_plot, '-', label='KF+DNN')
            plt.title(f"{sheet} Cycle {cycle} SoC")
            plt.xlabel("Time (s)")
            plt.ylabel("SoC")
            plt.legend(loc='lower right')
            plt.grid(linestyle=':')
            plt.tight_layout()
            plt.show()
            """
            plt.figure(figsize=(8, 4))
            sop_true_plot = SOP_true[start:]
            sop_kf_plot = SOP_kf[start:]
            plt.plot(t_plot, sop_true_plot, '-', label='SOP True')
            plt.plot(t_plot, sop_kf_plot, '--', label='SOP Pred')
            plt.ylabel('SOP (W)')
            plt.xlabel("Time (s)")
            plt.title(f"{sheet} Cycle {cycle} SOP")
            plt.grid(linestyle=':')
            plt.legend()
            plt.tight_layout()
            plt.show()


        print(f"{sheet} - Avg DNN RMSE: {np.mean(rmse_dnn_cycle):.8f}, "
              f"Avg KF+DNN RMSE: {np.mean(rmse_kf_cycle):.8f}, "
              f"Avg SOP RMSE: {np.mean(rmse_sop_cycle):.8f}")
