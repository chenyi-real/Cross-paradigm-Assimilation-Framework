import torch
import torch.nn as nn
from config import STATE_DIM, OBS_DIM, DEVICE, Q_CLASSICAL, R_CLASSICAL

class EquivalentCircuitModel(nn.Module):
    def __init__(self, R0=0.01, R1=0.015, C1=2000.0, ocv_slope=1.4, ocv_intercept=2.8):
        super().__init__()
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.ocv_slope = ocv_slope
        self.ocv_intercept = ocv_intercept

    def state_transition(self, soc, v_rc, current, dt, capacity):
        delta_soc = current * (dt / 3600.0) / capacity
        soc_pred = (soc + delta_soc).clamp(0., 1.)
        alpha = torch.exp(-dt / (self.R1 * self.C1))
        v_rc_pred = alpha * v_rc + (1 - alpha) * self.R1 * current
        return soc_pred, v_rc_pred

    def output_voltage(self, soc_pred, v_rc_pred, current):
        ocv = self.ocv_slope * soc_pred + self.ocv_intercept
        vt = ocv - self.R0 * current - v_rc_pred
        return vt

class KalmanFilterWrapperDNN(nn.Module):
    def __init__(self, kalman_net: nn.Module, ecm_params=None):
        super().__init__()
        self.kalman_net = kalman_net.to(DEVICE)
        self.ecm = EquivalentCircuitModel(**(ecm_params or {})).to(DEVICE)

        self.init_soc     = None
        self.use_dnn_init = False

    def forward(self,
                X_diff_batch: torch.Tensor,  # [B,T,3]
                Y_obs_batch:   torch.Tensor,  # [B,T]
                I_batch:       torch.Tensor,  # [B,T]
                dt_batch:      torch.Tensor,  # [B,T]
                capacity):                   # [B,1]
        B, T, _ = X_diff_batch.shape

        # 1) 自适应增益
        K_raw, _ = self.kalman_net(X_diff_batch)
        K_seq = torch.sigmoid(K_raw).reshape(B, T, STATE_DIM, OBS_DIM)

        # 2) 初始状态
        if self.use_dnn_init:
            assert self.init_soc is not None, "先给 kf_wrapper.init_soc 赋值"
            soc_est = self.init_soc
        else:
            vt0 = Y_obs_batch[:,0]
            i0  = I_batch[:,0]
            soc0 = (vt0 + self.ecm.R0*i0 - self.ecm.ocv_intercept) / self.ecm.ocv_slope
            soc_est = soc0.clamp(0.,1.).view(B,1)

        v_rc = torch.zeros((B,1), device=DEVICE)
        P_update = torch.eye(STATE_DIM, device=DEVICE).unsqueeze(0).expand(B, STATE_DIM, STATE_DIM)

        # H 矩阵
        H = torch.zeros((1, STATE_DIM), device=DEVICE)
        H[0,0] = 1.0
        H = H.unsqueeze(0).expand(B, 1, STATE_DIM)

        estimates = [soc_est]

        # 3) 滤波循环
        for t in range(1, T):
            it  = I_batch[:,t:t+1]
            dtt = dt_batch[:,t:t+1]

            # 状态预测
            soc_pred, v_rc_pred = self.ecm.state_transition(soc_est, v_rc, it, dtt, capacity)
            P_pred = P_update + Q_CLASSICAL

            # 观测预测
            zt = Y_obs_batch[:, t:t + 1].unsqueeze(-1)  # [B,1,1]
            innov = zt - soc_pred.unsqueeze(-1)  # soc_pred: [B,1]

            # 卡尔曼增益
            Kt = K_seq[:,t]  # [B,2,1]

            # 状态更新
            x_prior  = torch.cat([soc_pred, v_rc_pred], dim=1).unsqueeze(-1)  # [B,2,1]
            x_update = x_prior + torch.bmm(Kt, innov)                       # [B,2,1]
            x_update = x_update.squeeze(-1)                                 # [B,2]
            soc_est  = x_update[:,0:1].clamp(0.,1.)
            v_rc     = x_update[:,1:2]

            # 协方差更新
            KH       = torch.bmm(Kt, H)
            I_mat    = torch.eye(STATE_DIM, device=DEVICE).unsqueeze(0).expand(B, STATE_DIM, STATE_DIM)
            P_update = torch.bmm(I_mat - KH,
                                 torch.bmm(P_pred, (I_mat - KH).transpose(1,2))) \
                       + R_CLASSICAL * torch.bmm(Kt, Kt.transpose(1,2))

            estimates.append(soc_est)

        x_est_seq = torch.cat(estimates, dim=1)  # [B,T]
        return x_est_seq, None
