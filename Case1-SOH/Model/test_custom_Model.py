import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate

class SpikeMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_steps=4, dropout=0.2):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lif1 = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            threshold=1.0,
        )

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.lif2 = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            threshold=1.0,
        )

        self.dropout = nn.Dropout(dropout)
        # 残差分支，如果输入输出维度不一致
        self.downsample = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
        self.lif_res = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            threshold=1.0,
        )

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        out = self.fc1(x)
        out = self.bn1(out.view(B*S, -1)).view(B, S, -1)
        spk_rec1 = []
        for _ in range(self.num_steps):
            spk = self.lif1(out)
            spk_rec1.append(spk)
        spks1 = torch.stack(spk_rec1, dim=-1).mean(-1)

        out = self.fc2(spks1)
        out = self.bn2(out.view(B*S, -1)).view(B, S, -1)
        out = self.dropout(out)
        spk_rec2 = []
        for _ in range(self.num_steps):
            spk = self.lif2(out)
            spk_rec2.append(spk)
        spks2 = torch.stack(spk_rec2, dim=-1).mean(-1)

        # 残差连接
        if self.downsample is not None:
            res = self.downsample(x)
        else:
            res = x
        spk_rec3 = []
        for _ in range(self.num_steps):
            spk = self.lif_res(spks2 + res)
            spk_rec3.append(spk)
        out = torch.stack(spk_rec3, dim=-1).mean(-1)
        return out  # [B, S, hidden_dim]


class SpikeMLP(nn.Module):
    def __init__(self, input_dim=2, seq_len=12, hidden_dim=32, num_layers=3, num_steps=4, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(SpikeMLPBlock(
                input_dim=in_dim,
                hidden_dim=hidden_dim,
                num_steps=num_steps,
                dropout=dropout
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, S, D]  (512, 12, 2)
        out = x
        for block in self.net:
            out = block(out)  # [B, S, hidden_dim]
        return out  # (512, 12, 32)

# =================== 测试 ===================
if __name__ == "__main__":
    x = torch.randn(512, 12, 2)
    model = SpikeMLP(input_dim=2, seq_len=12, hidden_dim=32, num_layers=3, num_steps=4, dropout=0.2)
    y = model(x)
    print(y.shape)  # 应输出: torch.Size([512, 12, 32])
