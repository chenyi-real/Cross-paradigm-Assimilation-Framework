import math
import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from FreTS import Model as FreTSModel

class GEGLU(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2, bias=bias)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(b)

class ResGatedMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = GEGLU(dim, hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h

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

class TempConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1,
                 padding=None, pad_mode="replicate"):
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
        self.conv = nn.Conv1d(in_chan, out_chan,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=conv_padding)
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
                 conv_channels=(32, 64),
                 lstm_hidden=128,
                 lstm_layers=2,
                 bidirectional=True,
                 fc_hidden=64,
                 dropout=0.2,
                 return_feat=False):
        super().__init__()
        conv_layers = []
        in_ch = input_dim
        for out_ch in conv_channels:
            conv_layers.append(TempConvBlock(in_ch, out_ch, kernel_size=3))
            in_ch = out_ch
        self.conv_net = nn.Sequential(*conv_layers)

        self.dropout = nn.Dropout(dropout)
        self.lstm_input_size = conv_channels[-1]
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
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
        B, T, C = x.size()
        x = x.transpose(1, 2)
        x = self.conv_net(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        out, _ = self.lstm(x)

        scores = self.pool_attn(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        last = (out * weights).sum(dim=1)

        if self.return_feat:
            return last

        y = self.fc(last).view(B)
        return y


class TimeFreqBranch(nn.Module):
    def __init__(self,
                 seq_len: int = 16,
                 feature_size: int = 3,
                 safe_const_norm: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.safe_const_norm = bool(safe_const_norm)

        self.in_norm = nn.InstanceNorm1d(feature_size, affine=False, eps=1e-5)

        self.branch_1 = TimeSeriesModel(
            input_dim=feature_size,
            conv_channels=(32, 64),
            lstm_hidden=128,
            lstm_layers=2,
            bidirectional=True,
            fc_hidden=64,
            dropout=0.1,
            return_feat=True,
        )

        self.softmin = SoftExtrema(alpha=50.0, reduce="min")
        self.hint_gate = nn.Parameter(torch.tensor(0.30))
        feat_dim_branch1 = self.branch_1.lstm.hidden_size * self.branch_1.num_directions
        self.hint_proj = nn.Linear(3, feat_dim_branch1, bias=False)
        self.soc_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.kaiming_uniform_(self.hint_proj.weight, a=math.sqrt(5))

        hyperparams = {
            'embed_size': 128,
            'hidden_size': 128,
            'pre_length': seq_len,
            'feature_size': feature_size,
            'seq_length': seq_len,
            'channel_independence': 1,
            'sparsity_threshold': 0.01,
            'scale': 0.02,
        }
        self.branch_2 = FreTSModel(save_dir="fret_logs", **hyperparams)
        self.branch2_gate = nn.Parameter(torch.tensor(0.5))

        feat2_dim = seq_len * feature_size
        self.feat2_ln = nn.LayerNorm(feat2_dim)

        self.fc = nn.Sequential(
            nn.Linear(feat_dim_branch1 + feat2_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        B, T, C = x.shape
        raw = x

        xn = raw.transpose(1, 2)
        if not self.safe_const_norm:
            xn = self.in_norm(xn)
        else:
            var = xn.var(dim=-1, unbiased=False, keepdim=True)  # (B, C, 1)
            mean = xn.mean(dim=-1, keepdim=True)
            thr = 1e-6
            zn = (xn - mean) / torch.sqrt(var + 1e-5)
            const_mask = (var < thr).expand_as(xn)
            xn = torch.where(const_mask, xn, zn)
        xn = xn.transpose(1, 2)

        y1_feat = self.branch_1(xn)

        sig1 = raw[..., 2]
        sig2 = raw[..., 1]

        sig1_softmin = self.softmin(sig1, dim=1, keepdim=False)   # (B,)

        alpha = 50.0
        w_abs = torch.softmax(alpha * sig1.abs(), dim=1)
        i_softabsmax = (w_abs * sig1).sum(dim=1)

        sig2_softmin = self.softmin(sig2, dim=1, keepdim=False)
        sig20 = sig2[:, 0]
        sig2_drop = sig2_softmin - sig20

        hint_vec = torch.stack([sig1_softmin, i_softabsmax, sig2_drop], dim=1)
        y1_feat = y1_feat + self.hint_gate * self.hint_proj(hint_vec)

        y2_feat = self.branch_2(xn)
        y2_feat = self.feat2_ln(y2_feat)
        gate = torch.sigmoid(self.branch2_gate)
        y2_feat = gate * y2_feat

        feat = torch.cat((y1_feat, y2_feat), dim=1)
        y = self.fc(feat).view(B)
        return y


def timefreq_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    if y_pred.dim() > 1:
        y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    y_std = y_true.std(unbiased=False).clamp_min(5e-4)
    resid = (y_pred - y_true) / y_std
    data_loss = F.smooth_l1_loss(resid, torch.zeros_like(resid), beta=0.5)
    return data_loss

def _init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

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
                 hidden_dims=[256, 256], dropout=0.10,
                 return_feat=False,
                 patch_size: int = 4,
                 d_model: int = 256,
                 depth: int = 6,
                 token_mlp_ratio: float = 0.5,
                 channel_mlp_ratio: float = 4.0,
                 drop_path: float = 0.10,
                 use_in_norm: bool = True,
                 **kwargs):
        super().__init__()
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.return_feat = return_feat

        self.use_in_norm = use_in_norm  # ★ 记录开关
        if use_in_norm:
            self.in_norm = nn.InstanceNorm1d(self.input_dim, affine=True, eps=1e-5)
        else:
            self.in_norm = None
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
        _init_linear(self.hint_proj)
        self.hint_gate = nn.Parameter(torch.tensor(0.5))

        self.core_ln = nn.LayerNorm(d_model)
        self.hint_ln = nn.LayerNorm(64)

        self.film = nn.Sequential(
            nn.Linear(64, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 2 * d_model),
        )
        for m in self.film:
            if isinstance(m, nn.Linear):
                _init_linear(m)

        self.hint_to_core = nn.Linear(64, d_model, bias=False)
        _init_linear(self.hint_to_core)

        self.fuse_gate = nn.Sequential(
            nn.Linear(d_model + 64, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        for m in self.fuse_gate:
            if isinstance(m, nn.Linear):
                _init_linear(m)

        head_dim = d_model + 64
        n_blocks = max(2, len(hidden_dims))
        mlp_hidden = hidden_dims[0] if len(hidden_dims) > 0 else 256

        self.head_blocks = nn.ModuleList([
            ResGatedMLP(head_dim, mlp_hidden, dropout) for _ in range(n_blocks)
        ])

        self.out_ln = nn.LayerNorm(head_dim)
        self.out = nn.Linear(head_dim, 1)
        _init_linear(self.out)

        self.soc_scale = nn.Parameter(torch.tensor(1.0))

        nn.init.kaiming_uniform_(self.se_fc1.weight, a=math.sqrt(5)); nn.init.zeros_(self.se_fc1.bias)
        nn.init.kaiming_uniform_(self.se_fc2.weight, a=math.sqrt(5)); nn.init.zeros_(self.se_fc2.bias)

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

        if self.in_norm is not None:
            xn = self.in_norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            xn = x

        ch_mean = xn.mean(dim=1)
        gate = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(ch_mean))))
        xn = xn * gate.unsqueeze(1)

        z = self._patchify_embed(xn) + self.pos_embed

        for blk in self.blocks:
            z = blk(z)

        scores = self.attn_pool(z).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        feat_core = (z * weights).sum(dim=1)

        sig1 = x[..., 2]
        sig2 = x[..., 1]
        sig1_softmin = self.softmin(sig1, dim=1, keepdim=False)
        alpha = 50.0
        w_abs = torch.softmax(alpha * sig1.abs(), dim=1)
        i_softabsmax = (w_abs * sig1).sum(dim=1)
        sig2_softmin = self.softmin(sig2, dim=1, keepdim=False)
        sig20 = sig2[:, 0]
        sig2_drop = sig2_softmin - sig20

        hint = torch.stack([sig1_softmin, i_softabsmax, sig2_drop], dim=1)
        hint = self.hint_proj(hint) * torch.sigmoid(self.hint_gate)

        core = self.core_ln(feat_core)
        hint_n = self.hint_ln(hint)

        gamma_beta = self.film(hint_n)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = torch.tanh(gamma) * 0.5
        beta = torch.tanh(beta) * 0.5
        core_film = core * (1.0 + gamma) + beta

        hint_core = self.hint_to_core(hint_n)
        g = torch.sigmoid(self.fuse_gate(torch.cat([core_film, hint_n], dim=1)))
        core_fused = core_film + g * (hint_core - core_film)

        h = torch.cat([core_fused, hint_n], dim=1)

        for blk in self.head_blocks:
            h = blk(h)

        h = self.out_ln(h)
        y = self.out(h).view(B)
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
        use_in_norm: bool = True,
        **kwargs
    ):
        super().__init__()
        self.soc_scale = nn.Parameter(torch.tensor(1.0))
        self.use_in_norm = use_in_norm
        if use_in_norm:
            self.in_norm = nn.InstanceNorm1d(in_dim, affine=True, eps=1e-5)
        else:
            self.in_norm = None

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
        if self.in_norm is not None:
            z = self.in_norm(z)
        z = self.stem(z)
        z = self.backbone(z)

        gem = self.pool_gem(z)
        att = self.pool_attn(z)

        sig1 = x[..., 2]
        sig2 = x[..., 1]
        alpha = 40.0
        sig1_softmin = _softmin(sig1, alpha=alpha, dim=1)
        w_abs = torch.softmax(alpha * sig1.abs(), dim=1)
        sig1_softabsmax = (w_abs * sig1).sum(dim=1)
        sig2_softmin = _softmin(sig2, alpha=alpha, dim=1)
        sig20 = sig2[:, 0]
        sig2_drop = sig2_softmin - sig20

        hint = torch.stack([sig1_softmin, sig1_softabsmax, sig2_drop], dim=1)
        hint = self.hint_proj(hint)

        feat = torch.cat([gem, att, hint], dim=1)
        y = self.head(feat).squeeze(-1)
        return y

class LSTMRegressor(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 64, layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.10, proj_dim: int = 32,
                 use_in_norm: bool = True):
        super().__init__()
        self.soc_scale = nn.Parameter(torch.tensor(1.0))
        self.use_in_norm = use_in_norm
        if use_in_norm:
            self.in_norm = nn.InstanceNorm1d(in_dim, affine=True, eps=1e-5)
        else:
            self.in_norm = None

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
        if self.in_norm is not None:
            z = self.in_norm(z)
        z = z.transpose(1, 2)
        z = self.in_proj(z)
        z = self.drop(z)
        out, _ = self.lstm(z)
        scores  = self.pool_attn(out).squeeze(-1)
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
            use_in_norm: bool = True
    ):
        super().__init__()
        assert kernel_size >= 2 and kernel_size % 1 == 0
        self.soc_scale = nn.Parameter(torch.tensor(1.0))

        self.use_in_norm = use_in_norm
        if use_in_norm:
            self.in_norm = nn.InstanceNorm1d(in_dim, affine=True, eps=1e-5)
        else:
            self.in_norm = None

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
        if self.in_norm is not None:
            z = self.in_norm(z)
        z = self.tcn(z)
        scores = self.pool_attn(z).squeeze(1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(1)
        z_pool = (z * weights).sum(dim=-1)
        y = self.head(z_pool).squeeze(-1)
        return y