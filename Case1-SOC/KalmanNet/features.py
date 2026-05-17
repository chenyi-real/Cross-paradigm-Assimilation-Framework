import numpy as np
from config import Q_CLASSICAL, R_CLASSICAL

try:
    Q_SCALAR = float(Q_CLASSICAL.flatten()[0].item())
except Exception:
    Q_SCALAR = float(Q_CLASSICAL)
try:
    R_SCALAR = float(R_CLASSICAL.flatten()[0].item())
except Exception:
    R_SCALAR = float(R_CLASSICAL)

def compute_kalman_priors(traj):
    soc = traj['soc']  # shape (T,)
    T = soc.shape[0]

    x_pred      = np.zeros(T, dtype=float)
    x_update    = np.zeros(T, dtype=float)
    y_pred      = np.zeros(T, dtype=float)
    P_pred      = np.zeros(T, dtype=float)
    P_update    = np.zeros(T, dtype=float)
    K_classical = np.zeros(T, dtype=float)

    x_update[0]    = soc[0]
    P_update[0]    = 1.0
    y_pred[0]      = soc[0]
    x_pred[0]      = soc[0]
    K_classical[0] = 0.0

    for t in range(1, T):
        # —— 预测步 ——
        x_pred[t] = x_update[t - 1]
        P_pred[t] = P_update[t - 1] + Q_SCALAR
        y_pred[t] = x_pred[t]

        # —— 更新步 ——
        K_classical[t] = P_pred[t] / (P_pred[t] + R_SCALAR)
        x_update[t]    = x_pred[t] + K_classical[t] * (soc[t] - y_pred[t])
        P_update[t]    = (1.0 - K_classical[t]) * P_pred[t]

    return {
        'x_pred':      x_pred,      # shape (T,)
        'y_pred':      y_pred,      # shape (T,)
        'x_update':    x_update,    # shape (T,)
        'K_classical': K_classical  # shape (T,)
    }


def compute_F1_F2_F4(traj, kalman_priors):
    soc      = traj['soc']
    x_pred   = kalman_priors['x_pred']
    y_pred   = kalman_priors['y_pred']
    x_update = kalman_priors['x_update']
    T        = soc.shape[0]

    F1 = np.zeros((T, 1), dtype=float)
    F2 = np.zeros((T, 1), dtype=float)
    F4 = np.zeros((T, 1), dtype=float)

    for t in range(T):
        F1[t, 0] = soc[t] - soc[t-1] if t > 0 else 0.0
        F2[t, 0] = soc[t] - y_pred[t]
        F4[t, 0] = x_update[t] - x_pred[t]

    return F1, F2, F4


def build_feature_sequence(traj):
    priors = compute_kalman_priors(traj)
    F1, F2, F4 = compute_F1_F2_F4(traj, priors)
    X_diff = np.concatenate([F1, F2, F4], axis=1)  # (T,3)
    y_true = traj['soc']                           # (T,)
    return X_diff, y_true
