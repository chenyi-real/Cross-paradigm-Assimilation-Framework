import torch
import torch.nn as nn
import torch.nn.functional as F
from config import STATE_DIM, OBS_DIM, INPUT_DIM, HIDDEN_DIM

class KalmanNetNet1(nn.Module):
    def __init__(self):
        super().__init__()
        m, n = STATE_DIM, OBS_DIM
        self.h_dim = HIDDEN_DIM

        self.fc_in = nn.Linear(INPUT_DIM, self.h_dim)
        self.layer_norm = nn.LayerNorm(self.h_dim)

        self.gru = nn.GRU(self.h_dim, self.h_dim, batch_first=True)

        self.fc_out = nn.Linear(self.h_dim, m * n)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.zeros_(self.fc_in.bias)
        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.1)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, X_diff_seq, h_0=None):
        x = self.fc_in(X_diff_seq)            # [B, T, h_dim]
        x = self.layer_norm(x)
        x = F.relu(x)

        with torch.backends.cudnn.flags(enabled=False):
            out, h_n = self.gru(x, h_0)       # out: [B, T, h_dim]

        K_vec = self.fc_out(out)              # [B, T, m*n]
        B, T, _ = K_vec.size()

        K_seq = K_vec.reshape(B, T, STATE_DIM, OBS_DIM)

        return K_seq, h_n

class CNNGRUAttention(nn.Module):
    def __init__(self,
                 in_feats: int = 2,
                 cnn_channels: int = 16,
                 kernel_size: int = 3,
                 pool_size: int = 2,
                 gru_hidden: int = 32,
                 out_feats: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_feats, cnn_channels, kernel_size,
                              padding=kernel_size//2)
        self.pool = nn.MaxPool1d(pool_size)
        self.gru = nn.GRU(cnn_channels, gru_hidden, batch_first=True)
        self.attn = nn.Linear(gru_hidden, 1)
        self.fc = nn.Linear(gru_hidden, out_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = self.pool(x)                 # (B, C, T')
        # → (B, T', C)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)             # out: (B, T', H)

        alpha = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        context = (out * alpha.unsqueeze(-1)).sum(dim=1)
        soc = torch.sigmoid(self.fc(context))
        return soc


