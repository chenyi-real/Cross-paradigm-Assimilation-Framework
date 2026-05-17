import math
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
import os
from torch.nn.utils import weight_norm
import FreTS
from config import EXP_DIR, DISTILL_ENABLE, WINDOW_NASA, WINDOW_CALCE, WINDOW_MIT, WINDOW_UDDS, DATASET_NAME, \
    WHOLE_ENABLE, WHOLE_ARCH, WHOLE_A_CKPT

def _init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class SoftExtrema(nn.Module):
    def __init__(self, alpha: float = 50.0, reduce: str = "min"):
        super().__init__()
        assert reduce in ("min", "max")
        self.alpha = alpha
        self.reduce = reduce

    def forward(self, x: torch.Tensor, dim: int = -1, keepdim: bool = False):
        a = self.alpha
        if self.reduce == "min":
            y = -torch.logsumexp(-a * x, dim=dim, keepdim=keepdim) / a
        else:
            y = torch.logsumexp(a * x, dim=dim, keepdim=keepdim) / a
        return y

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

class MixerBlock(nn.Module):
    def __init__(self, n_tokens: int, dim: int,
                 token_mlp_ratio: float = 0.5,
                 channel_mlp_ratio: float = 4.0,
                 drop: float = 0.10, drop_path: float = 0.10):
        super().__init__()
        t_hidden = max(16, int(n_tokens * token_mlp_ratio))
        c_hidden = max(64, int(dim * channel_mlp_ratio))

        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, t_hidden, bias=True),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(t_hidden, n_tokens, bias=True),
            nn.Dropout(drop),
        )
        self.dp1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, c_hidden, bias=True),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(c_hidden, dim, bias=True),
            nn.Dropout(drop),
        )
        self.dp2 = DropPath(drop_path)

        self.apply(_init_linear)

    def forward(self, x):
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + self.dp1(y)

        y2 = self.channel_mlp(self.norm2(x))
        x = x + self.dp2(y2)
        return x

class MLPRegressor(nn.Module):
    def __init__(self,
                 input_dim=3, seq_len=64,
                 hidden_dims=None, dropout=0.10,
                 return_feat=False,
                 patch_size: int = 4,
                 d_model: int = 256,
                 depth: int = 6,
                 token_mlp_ratio: float = 0.5,
                 channel_mlp_ratio: float = 4.0,
                 drop_path: float = 0.10,
                 **kwargs):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.input_dim = int(input_dim)
        if self.input_dim == 2:
            self.soc_idx, self.i_idx = 0, 1
        else:
            self.soc_idx, self.i_idx = 1, 2
        self.seq_len = int(seq_len)
        self.return_feat = return_feat

        self.in_norm = nn.InstanceNorm1d(self.input_dim, affine=True, eps=1e-5)
        se_hidden = max(8, self.input_dim * 2)
        self.se_fc1 = nn.Linear(self.input_dim, se_hidden, bias=True)
        self.se_fc2 = nn.Linear(se_hidden, self.input_dim, bias=True)

        self.patch_size = max(1, int(patch_size))
        self.n_tokens = int(math.ceil(self.seq_len / self.patch_size))
        self.patch_embed = nn.Linear(self.input_dim * self.patch_size, d_model, bias=True)
        _init_linear(self.patch_embed)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dpr_list = torch.linspace(0, drop_path, steps=max(1, depth)).tolist()
        self.blocks = nn.ModuleList([
            MixerBlock(self.n_tokens, d_model,
                       token_mlp_ratio=token_mlp_ratio,
                       channel_mlp_ratio=channel_mlp_ratio,
                       drop=dropout, drop_path=dpr_list[i])
            for i in range(depth)
        ])

        self.attn_pool = nn.Linear(d_model, 1, bias=False)
        _init_linear(self.attn_pool)

        self.softmin = SoftExtrema(alpha=50.0, reduce="min")
        self.hint_proj = nn.Linear(3, 64, bias=False)
        self.hint_gate = nn.Parameter(torch.tensor(0.5))
        _init_linear(self.hint_proj)

        head_in = d_model + 64
        head_layers = []
        dim = head_in
        for h in hidden_dims:
            head_layers += [nn.Linear(dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            dim = h
        head_layers += [nn.Linear(dim, 1)]
        self.head = nn.Sequential(*head_layers)

        self.soc_scale = nn.Parameter(torch.tensor(1.0))

        for m in self.head:
            if isinstance(m, nn.Linear):
                _init_linear(m)
        nn.init.kaiming_uniform_(self.se_fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.se_fc1.bias)
        nn.init.kaiming_uniform_(self.se_fc2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.se_fc2.bias)

    def _patchify_embed(self, x_btC: torch.Tensor) -> torch.Tensor:
        B, T, C = x_btC.shape
        P = self.patch_size
        N = self.n_tokens
        need = N * P - T
        if need > 0:
            tail = x_btC[:, -1:, :].expand(B, need, C)
            x_btC = torch.cat([x_btC, tail], dim=1)
        x = x_btC.view(B, N, P, C).reshape(B, N, P * C)
        z = self.patch_embed(x)
        return z

    def forward(self, x):
        B, T, D = x.size()

        if T != self.seq_len:
            x = F.interpolate(x.transpose(1, 2), size=self.seq_len, mode='linear', align_corners=False).transpose(1, 2)

        xn = self.in_norm(x.transpose(1, 2)).transpose(1, 2)
        ch_mean = xn.mean(dim=1)
        gate = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(ch_mean))))
        xn = xn * gate.unsqueeze(1)

        z = self._patchify_embed(xn) + self.pos_embed

        for blk in self.blocks:
            z = blk(z)

        scores = self.attn_pool(z).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        feat_core = (z * weights).sum(dim=1)

        i = x[..., self.i_idx]
        soc = x[..., self.soc_idx]
        i_softmin = self.softmin(i, dim=1, keepdim=False)
        alpha = 50.0
        w_abs = torch.softmax(alpha * i.abs(), dim=1)
        i_softabsmax = (w_abs * i).sum(dim=1)
        soc_softmin = self.softmin(soc, dim=1, keepdim=False)
        soc0 = soc[:, 0]
        soc_drop = soc_softmin - soc0

        hint = torch.stack([i_softmin, i_softabsmax, soc_drop], dim=1)
        hint = self.hint_proj(hint) * torch.sigmoid(self.hint_gate)

        h = torch.cat([feat_core, hint], dim=1)
        for layer in self.head[:-1]:
            h = layer(h)
        feat = h
        if self.return_feat:
            return feat
        y = self.head[-1](feat).view(B)
        return y

class _GeM1d(nn.Module):
    def __init__(self, p: float = 1.3, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps)
        p = torch.clamp(self.p, 1.0, 6.0)
        x = x.pow(p).mean(dim=-1)
        return x.pow(1.0 / p)

class SE1d(nn.Module):
    def __init__(self, channels: int, r: int = 8, min_ch: int = 8):
        super().__init__()
        hidden = max(min_ch, channels // r)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        z = self.pool(x).squeeze(-1)
        z = F.relu(self.fc1(z), inplace=True)
        z = torch.sigmoid(self.fc2(z)).unsqueeze(-1)
        return x * z

class _MSDepthwise(nn.Module):
    def __init__(self, c_in, c_out, ks=(3, 5, 7), dilations=(1, 2, 4), drop=0.10):
        super().__init__()
        assert len(ks) == len(dilations)
        self.branches = nn.ModuleList()
        for k, d in zip(ks, dilations):
            pad = (k - 1) // 2 * d
            conv = nn.Conv1d(
                c_in, c_in, kernel_size=k, padding=pad, dilation=d,
                groups=c_in, bias=False
            )
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
            self.branches.append(nn.Sequential(
                conv, nn.BatchNorm1d(c_in), nn.ReLU(inplace=True)
            ))
        self.mix = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.mix.weight, a=math.sqrt(5))
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.se = SE1d(c_out, r=8)

        self.proj = None
        if c_in != c_out:
            self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

    def forward(self, x):
        ys = [b(x) for b in self.branches]
        y = torch.stack(ys, dim=0).sum(dim=0)
        y = self.act(self.bn(self.mix(y)))
        y = self.se(y)
        y = self.drop(y)
        res = x if self.proj is None else self.proj(x)
        return y + res

class _AttnPool1d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.score = nn.Conv1d(c, 1, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.score.weight, a=math.sqrt(5))

    def forward(self, z):
        s = self.score(z).squeeze(1)
        w = torch.softmax(s, dim=-1).unsqueeze(1)
        return (z * w).sum(dim=-1)


def _softmin(x: torch.Tensor, alpha: float = 40.0, dim: int = 1):
    w = torch.softmax(-alpha * x, dim=dim)
    return (w * x).sum(dim=dim)

class CNNRegressor(nn.Module):
    def __init__(
            self,
            in_dim: int = 3,
            channels=(64, 96, 128),
            dropout: float = 0.10,
            kernel_size: int = 5,
            ks=None,
            dilations=(1, 2, 4),
    ):
        super().__init__()
        self.soc_scale = nn.Parameter(torch.tensor(1.0))
        self.in_norm = nn.InstanceNorm1d(in_dim, affine=True, eps=1e-5)

        if in_dim == 3:
            self.soc_idx, self.i_idx = 1, 2
        else:
            self.soc_idx, self.i_idx = 0, 1

        if ks is None:
            def _fix(k: int):
                k = int(k)
                if k < 3: k = 3
                if k % 2 == 0: k += 1
                return k

            ks_list = tuple(sorted({_fix(kernel_size - 2), _fix(kernel_size), _fix(kernel_size + 2)}))
        else:
            ks_list = tuple(int(x) for x in ks)

        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, channels[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_uniform_(self.stem[0].weight, a=math.sqrt(5))

        blocks = []
        c_in = channels[0]
        for c_out in channels[1:]:
            blocks.append(_MSDepthwise(c_in, c_out, ks=ks_list, dilations=dilations, drop=dropout))
            c_in = c_out
        self.backbone = nn.Sequential(*blocks)

        self.pool_gem = _GeM1d(p=1.6)
        self.pool_attn = _AttnPool1d(c_in)

        self.hint_proj = nn.Linear(3, 64, bias=False)
        nn.init.kaiming_uniform_(self.hint_proj.weight, a=math.sqrt(5))

        head_in = c_in * 2 + 64
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        z = x.transpose(1, 2)
        z = self.in_norm(z)
        z = self.stem(z)
        z = self.backbone(z)

        gem = self.pool_gem(z)
        att = self.pool_attn(z)

        i = x[..., self.i_idx]
        soc = x[..., self.soc_idx]
        alpha = 40.0
        i_softmin = _softmin(i, alpha=alpha, dim=1)
        w_abs = torch.softmax(alpha * i.abs(), dim=1)
        i_softabsmax = (w_abs * i).sum(dim=1)
        soc_softmin = _softmin(soc, alpha=alpha, dim=1)
        soc0 = soc[:, 0]
        soc_drop = soc_softmin - soc0

        hint = torch.stack([i_softmin, i_softabsmax, soc_drop], dim=1)
        hint = self.hint_proj(hint)

        feat = torch.cat([gem, att, hint], dim=1)
        y = self.head(feat).squeeze(-1)
        return y

class LSTMRegressor(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 64, layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.10, proj_dim: int = 32):
        super().__init__()
        self.soc_scale = nn.Parameter(torch.tensor(1.0))
        self.in_norm = nn.InstanceNorm1d(in_dim, affine=True, eps=1e-5)

        self.in_proj = nn.Linear(in_dim, proj_dim, bias=False)
        _init_linear(self.in_proj)

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0
        )
        self.num_directions = 2 if bidirectional else 1
        self.pool_attn = nn.Linear(hidden * self.num_directions, 1, bias=False)
        _init_linear(self.pool_attn)
        self.drop = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden * self.num_directions, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        for m in self.fc:
            if isinstance(m, nn.Linear):
                _init_linear(m)

    def forward(self, x: torch.Tensor):
        z = x.transpose(1, 2)
        z = self.in_norm(z).transpose(1, 2)
        z = self.in_proj(z)
        z = self.drop(z)
        out, _ = self.lstm(z)
        scores = self.pool_attn(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        h = (out * weights).sum(dim=1)
        y = self.fc(h).squeeze(-1)
        return y

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size: int, dilation: int = 1, bias: bool = True,
                 use_weight_norm: bool = True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, bias=bias)
        if use_weight_norm:
            conv = weight_norm(conv)
        self.net = nn.Sequential(nn.ConstantPad1d((self.pad, 0), 0.0), conv)

        if isinstance(conv, nn.Conv1d):
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x):
        return self.net(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size: int, dilation: int, dropout: float = 0.1,
                 use_weight_norm: bool = True):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation, bias=True,
                                  use_weight_norm=use_weight_norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation, bias=True,
                                  use_weight_norm=use_weight_norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = None
        if in_ch != out_ch:
            ds = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
            nn.init.kaiming_uniform_(ds.weight, a=math.sqrt(5))
            self.downsample = ds

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.relu2(y)
        y = self.drop2(y)

        res = x if self.downsample is None else self.downsample(x)
        return y + res

class TCNRegressor(nn.Module):
    def __init__(
            self,
            in_dim: int = 3,
            channels=(64, 64, 64, 64),
            kernel_size: int = 3,
            dropout: float = 0.10,
            use_weight_norm: bool = True,
    ):
        super().__init__()
        assert kernel_size >= 2 and kernel_size % 1 == 0
        self.soc_scale = nn.Parameter(torch.tensor(1.0))

        self.in_norm = nn.InstanceNorm1d(in_dim, affine=True, eps=1e-5)

        c_in = in_dim
        blocks = []
        for i, c_out in enumerate(channels):
            dilation = 2 ** i
            blocks.append(
                TemporalBlock(
                    in_ch=c_in, out_ch=c_out,
                    kernel_size=kernel_size, dilation=dilation,
                    dropout=dropout, use_weight_norm=use_weight_norm
                )
            )
            c_in = c_out
        self.tcn = nn.Sequential(*blocks)

        self.pool_attn = nn.Conv1d(c_in, 1, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.pool_attn.weight, a=math.sqrt(5))

        self.head = nn.Linear(c_in, 1, bias=True)
        _init_linear(self.head)

    def forward(self, x: torch.Tensor):
        z = x.transpose(1, 2)
        z = self.in_norm(z)
        z = self.tcn(z)
        scores = self.pool_attn(z).squeeze(1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(1)
        z_pool = (z * weights).sum(dim=-1)
        y = self.head(z_pool).squeeze(-1)
        return y

class TempConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=None, pad_mode="replicate"):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.pad = None
        if padding > 0 and pad_mode in ("replicate", "reflection"):
            Pad = nn.ReplicationPad1d if pad_mode == "replicate" else nn.ReflectionPad1d
            self.pad = Pad((padding, padding))
            conv_padding = 0
        else:
            conv_padding = padding
        self.conv = nn.Conv1d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=conv_padding)
        self.bn = nn.BatchNorm1d(out_chan)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class TimeSeriesModel(nn.Module):
    def __init__(self,
                 input_dim=3,
                 conv_channels=None,
                 lstm_hidden=128,
                 lstm_layers=2,
                 bidirectional=True,
                 fc_hidden=64,
                 dropout=0.2,
                 return_feat=False):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]
        conv_layers = []
        in_ch = input_dim
        for out_ch in conv_channels:
            conv_layers.append(TempConvBlock(in_ch, out_ch, kernel_size=3))
            in_ch = out_ch
        self.conv_net = nn.Sequential(*conv_layers)

        self.dropout = nn.Dropout(dropout)
        self.lstm_input_size = conv_channels[-1]
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.num_directions = 2 if bidirectional else 1
        self.pool_attn = nn.Linear(lstm_hidden * self.num_directions, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * self.num_directions, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )
        self.return_feat = return_feat

    def forward(self, x):
        batch = x.size(0)
        seq_len = x.size(1)
        x = x.transpose(1, 2)
        x = self.conv_net(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        out, (h_n, c_n) = self.lstm(x)
        scores = self.pool_attn(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        last = torch.sum(out * weights.unsqueeze(-1), dim=1)
        if self.return_feat:
            return last
        y = self.fc(last)
        y = y.view(batch)
        return y

class TwoBranchFramework(nn.Module):
    def __init__(self,
                 ):
        super().__init__()
        is_calce_voltage = (str(DATASET_NAME).upper() == "CALCE") and (not WHOLE_ENABLE) and (not DISTILL_ENABLE)
        self.in_dim = 2 if is_calce_voltage else 3
        self.in_norm = nn.InstanceNorm1d(self.in_dim, affine=False, eps=1e-5)

        if self.in_dim == 2:
            self.soc_idx, self.i_idx = 0, 1
        else:
            self.soc_idx, self.i_idx = 1, 2

        self.branch_1 = TimeSeriesModel(input_dim=self.in_dim,
                                        conv_channels=[32, 64],
                                        lstm_hidden=128,
                                        lstm_layers=2,
                                        bidirectional=True,
                                        fc_hidden=64,
                                        dropout=0.1,
                                        return_feat=True)
        if WHOLE_ENABLE:
            pre_len = WINDOW_UDDS
        else:
            if DATASET_NAME == "NASA":
                pre_len = WINDOW_NASA
            elif DATASET_NAME == "MIT":
                pre_len = WINDOW_MIT
            else:
                pre_len = WINDOW_CALCE

        hyperparameters = {
            'embed_size': 128,
            'hidden_size': 128,
            'pre_length': pre_len,
            'feature_size': self.in_dim,
            'seq_length': pre_len,
            'channel_independence': 1,
            'sparsity_threshold': 0.01,
            'scale': 0.02,
        }

        self.softmin = SoftExtrema(alpha=50.0, reduce="min")
        self.hint_gate = nn.Parameter(torch.tensor(0.30))
        self.hint_proj = nn.Linear(3, 256, bias=False)
        self.soc_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.kaiming_uniform_(self.hint_proj.weight, a=math.sqrt(5))

        self.branch_2 = FreTS.Model(EXP_DIR, **hyperparameters)
        self.branch2_gate = nn.Parameter(torch.tensor(0.5))

        feat2_dim = hyperparameters['pre_length'] * hyperparameters['feature_size']
        self.feat2_ln = nn.LayerNorm(feat2_dim)
        self.fc = nn.Sequential(
            nn.Linear(256 + feat2_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        B, T, _ = x.shape
        raw = x
        if str(DATASET_NAME).upper() == "CALCE" and (not WHOLE_ENABLE) and (not DISTILL_ENABLE):
            xn = raw
        else:
            xn = raw.transpose(1, 2)
            xn = self.in_norm(xn)
            xn = xn.transpose(1, 2)

        y1_feat = self.branch_1(xn)

        i_softmin = self.softmin(raw[..., self.i_idx], dim=1, keepdim=False)

        alpha = 50.0
        i = raw[..., self.i_idx]
        w_abs = torch.softmax(alpha * i.abs(), dim=1)
        i_softabsmax = (w_abs * i).sum(dim=1)

        soc_softmin = self.softmin(raw[..., self.soc_idx], dim=1, keepdim=False)
        soc0 = raw[:, 0, self.soc_idx]
        soc_drop = soc_softmin - soc0

        hint_vec = torch.stack([i_softmin, i_softabsmax, soc_drop], dim=1)
        y1_feat = y1_feat + self.hint_gate * self.hint_proj(hint_vec)

        y2_feat = self.branch_2(xn)
        y2_feat = self.feat2_ln(y2_feat)
        gate = torch.sigmoid(self.branch2_gate)
        y_feat = torch.cat((y1_feat, gate * y2_feat), dim=1)
        y = self.fc(y_feat).view(B)
        return y

class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=128, drop=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1, bias=False),
        )
        self.apply(_init_linear)

    def forward(self, x):
        return self.net(x).squeeze(-1)

class STARFusion(nn.Module):
    def __init__(self, dimA: int, dimB: int, core_dim: int = 128, drop: float = 0.10,
                 d_series: Optional[int] = None):
        super().__init__()
        self.dimA = dimA
        self.dimB = dimB
        self.core_dim = core_dim
        self.d_series = d_series if d_series is not None else min(dimA, dimB)

        self.projA = nn.Linear(dimA, self.d_series, bias=False)
        self.projB = nn.Linear(dimB, self.d_series, bias=False)

        self.gen1 = nn.Linear(self.d_series, self.d_series)
        self.gen2 = nn.Linear(self.d_series, self.core_dim)
        self.gen3 = nn.Linear(self.d_series + self.core_dim, self.d_series)
        self.gen4 = nn.Linear(self.d_series, self.d_series)

        self.outA = nn.Linear(self.d_series, dimA, bias=False)
        self.outB = nn.Linear(self.d_series, dimB, bias=False)
        self.lnA = nn.LayerNorm(dimA)
        self.lnB = nn.LayerNorm(dimB)
        self.drop = nn.Dropout(drop)

        self.headA = nn.Linear(dimA, 1, bias=False)
        self.headB = nn.Linear(dimB, 1, bias=False)

    def _star_core(self, x: torch.Tensor):
        B, C, d_series = x.shape
        combined_mean = F.gelu(self.gen1(x))
        combined_mean = self.gen2(combined_mean)

        weight = F.softmax(combined_mean, dim=1)
        if self.training:
            prob = weight.mean(-1)
            idx = torch.multinomial(prob, 1)
            combined_mean = combined_mean.gather(
                1, idx.unsqueeze(-1).expand(-1, 1, self.core_dim)
            )
        else:
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True)

        if combined_mean.shape[1] != C:
            combined_mean = combined_mean.repeat(1, C, 1)

        if combined_mean.shape[-1] != self.core_dim:
            Bc, Cc, L = combined_mean.shape
            combined_mean = F.interpolate(
                combined_mean.reshape(Bc * Cc, 1, L),
                size=self.core_dim,
                mode='linear',
                align_corners=False
            ).reshape(Bc, Cc, self.core_dim)

        combined_mean_cat = torch.cat([x, combined_mean], dim=-1)
        assert combined_mean_cat.shape[-1] == self.d_series + self.core_dim, \
            f"got {combined_mean_cat.shape[-1]}, expect {self.d_series + self.core_dim}"
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        y = self.gen4(combined_mean_cat)
        return y

    def forward(self, hA: torch.Tensor, hB: torch.Tensor):
        a = self.projA(hA)
        b = self.projB(hB)
        x = torch.stack([a, b], dim=1)

        y = self._star_core(x)

        yA = y[:, 0, :]
        yB = y[:, 1, :]
        hA2 = self.lnA(self.drop(self.outA(yA) + hA))
        hB2 = self.lnB(self.drop(self.outB(yB) + hB))

        partA = self.headA(hA2).squeeze(-1)
        partB = self.headB(hB2).squeeze(-1)
        return partA, partB, hA2, hB2

class RegressorAsEncoder(nn.Module):
    def __init__(self, regressor: nn.Module):
        super().__init__()
        self.backbone = regressor
        self._feat = None

        last_lin = None
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                last_lin = m

        self.out_dim = int(last_lin.in_features)

        def _pre_hook(m, inputs):
            x = inputs[0]
            self._feat = x

        self._hook = last_lin.register_forward_pre_hook(_pre_hook)

    def forward(self, x, *args, **kwargs):
        self._feat = None
        _ = self.backbone(x, *args, **kwargs)
        if self._feat is None:
            raise RuntimeError("[RegressorAsEncoder] 未捕获到特征")
        f = self._feat
        if f.dim() > 2:
            f = f.reshape(f.size(0), -1)
        return f

class WholeNet(nn.Module):
    def __init__(self, encoder_a: nn.Module, encoder_b: nn.Module, dimA: int, dimB: int,
                 head_hidden: int = 128, drop: float = 0.10):
        super().__init__()
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.star = STARFusion(dimA=dimA, dimB=dimB, core_dim=128, drop=drop)
        self.head = MLPHead(in_dim=dimA + dimB, hidden=head_hidden, drop=drop)

        for p in self.encoder_a.parameters():
            p.requires_grad = False

    def forward(self, xa: torch.Tensor, xb: torch.Tensor):
        hA = self.encoder_a(xa)
        hB = self.encoder_b(xb)

        _, _, hA2, hB2 = self.star(hA, hB)
        y = self.head(torch.cat([hA2, hB2], dim=-1))
        return y.squeeze(-1), (hA2, hB2)

def build_whole_model():
    arch = str(WHOLE_ARCH).lower()
    ckpt_path = WHOLE_A_CKPT

    if arch == "two_branch":
        A_raw = TwoBranchFramework()
        B_raw = TwoBranchFramework()
    elif arch == "cnn":
        A_raw = CNNRegressor(in_dim=3, channels=(32, 64, 128), kernel_size=5, dropout=0.10)
        B_raw = CNNRegressor(in_dim=3, channels=(32, 64, 128), kernel_size=5, dropout=0.10)
    elif arch == "lstm":
        A_raw = LSTMRegressor(in_dim=3, hidden=128, layers=2, bidirectional=True, dropout=0.10)
        B_raw = LSTMRegressor(in_dim=3, hidden=128, layers=2, bidirectional=True, dropout=0.10)
    elif arch == "mlp":
        A_raw = MLPRegressor(input_dim=3, seq_len=WINDOW_UDDS, hidden_dims=[256, 256], dropout=0.10)
        B_raw = MLPRegressor(input_dim=3, seq_len=WINDOW_UDDS, hidden_dims=[256, 256], dropout=0.10)
    elif arch == "tcn":
        A_raw = TCNRegressor(in_dim=3, channels=(64, 64, 64, 64), kernel_size=3, dropout=0.10)
        B_raw = TCNRegressor(in_dim=3, channels=(64, 64, 64, 64), kernel_size=3, dropout=0.10)
    elif arch == "student":
        A_raw = StudentMLPRegressor(
            in_dim=3,
            seq_len=WINDOW_UDDS,
            hidden_size=256,
            d_model_s=64,
            e_layers=2,
            down_layers=1,
            down_window=2,
            dropout=0.10,
            head_hidden=128,
        )
        B_raw = StudentMLPRegressor(
            in_dim=3,
            seq_len=WINDOW_UDDS,
            hidden_size=256,
            d_model_s=64,
            e_layers=2,
            down_layers=1,
            down_window=2,
            dropout=0.10,
            head_hidden=128,
        )
    else:
        raise ValueError(f"未知 WHOLE_ARCH: {arch}")

    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            A_raw.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
            print(f"[WholeNet] A({arch}) loaded from {ckpt_path}")
        except Exception as e:
            print(f"[WholeNet] A({arch}) load failed: {e}")

    encA = RegressorAsEncoder(A_raw)
    encB = RegressorAsEncoder(B_raw)
    dimA = encA.out_dim
    dimB = encB.out_dim

    model = WholeNet(encA, encB, dimA, dimB, head_hidden=128, drop=0.10)
    return model

class MLPStudentEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int = 256,
                 d_model_s: int = 64, e_layers: int = 2,
                 down_layers: int = 1, down_window: int = 2,
                 dropout: float = 0.10):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.d_model_s = d_model_s
        self.down_layers = down_layers
        self.down_window = down_window
        self.in_proj = nn.Linear(in_dim, d_model_s, bias=False)
        self.pw = nn.Sequential(
            *sum([[nn.Linear(d_model_s, d_model_s), nn.GELU(), nn.Dropout(dropout)] for _ in range(e_layers)], []))
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model_s, hidden_size, bias=False)
        self.out_ln = nn.LayerNorm(hidden_size)
        self.apply(_init_linear)

    def _multi_scale_inputs(self, x):
        xs = [x]
        for _ in range(self.down_layers):
            last = xs[-1].permute(0, 2, 1)
            ds = nn.AvgPool1d(self.down_window, ceil_mode=True)(last).transpose(1, 2)
            xs.append(ds)
        return xs

    def forward_with_feats(self, x: torch.Tensor, lengths=None):
        if lengths is not None:
            Lmax = int(lengths.max().item())
            x = x[:, :Lmax]
        xs = self._multi_scale_inputs(x)
        feats = []
        for xi in xs:
            z = self.in_proj(xi)
            z = self.pw(z)
            feats.append(z)
        B, T0, _ = feats[0].shape
        if lengths is None:
            pooled = feats[0].mean(dim=1)
        else:
            device = x.device
            t = torch.arange(T0, device=device).unsqueeze(0).expand(B, T0)
            m = (t < lengths.unsqueeze(1)).unsqueeze(-1).float()
            pooled = (feats[0] * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        h = self.out_ln(self.drop(self.out_proj(pooled)))
        return h, feats

    def forward(self, x: torch.Tensor, lengths=None):
        h, _ = self.forward_with_feats(x, lengths)
        return h

class StudentMLPRegressor(nn.Module):
    def __init__(
            self,
            in_dim: int = 3,
            seq_len: int = 64,
            hidden_size: int = 256,
            d_model_s: int = 64,
            e_layers: int = 2,
            down_layers: int = 1,
            down_window: int = 2,
            dropout: float = 0.10,
            head_hidden: int = 128,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.seq_len = int(seq_len)

        self.soc_scale = nn.Parameter(torch.tensor(1.0))

        self.encoder = MLPStudentEncoder(
            in_dim=self.in_dim,
            hidden_size=int(hidden_size),
            d_model_s=int(d_model_s),
            e_layers=int(e_layers),
            down_layers=int(down_layers),
            down_window=int(down_window),
            dropout=float(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(int(hidden_size), int(head_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden), 1),
        )
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"[StudentMLPRegressor] expected 3D input (B,T,C), got {tuple(x.shape)}")

        if x.size(1) != self.seq_len:
            x = F.interpolate(
                x.transpose(1, 2),
                size=self.seq_len,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)

        h = self.encoder(x)
        y = self.head(h).squeeze(-1)
        return y
