import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

def _init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Predictor(nn.Module):
    def __init__(self,input_dim=40):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim,32),
            Sin(),
            nn.Linear(32,1)
        )
        self.input_dim = input_dim
    def forward(self,x):
        return self.net(x) 


class SOH_StudentMLP(nn.Module):
    """
    学生 MLP 模型：
    - 内部自带 encoder 结构（不再依赖外部 Encoder 类）
    - encoder: 多层 Linear + Sin + Dropout，输出 feature
    - predictor: 从 feature -> 预测值
    - forward 返回 (pred, feat)
    """
    def __init__(
        self,
        input_dim=13,
        feature_dim=32,
        layers_num=3,
        hidden_dim=60,
        dropout=0.2,
    ):
        super(SOH_StudentMLP, self).__init__()

        assert layers_num >= 2, "layers_num must be >= 2"

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        # -------------------- 构建 encoder --------------------
        layers = []
        for i in range(layers_num):
            if i == 0:
                # 输入层
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(Sin())
            elif i == layers_num - 1:
                # 输出到特征维度
                layers.append(nn.Linear(hidden_dim, feature_dim))
            else:
                # 中间层
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(Sin())
                layers.append(nn.Dropout(p=dropout))

        self.encoder = nn.Sequential(*layers)
        self._init_encoder_weights()

        # -------------------- predictor --------------------
        self.predictor = Predictor(input_dim=feature_dim)

    def _init_encoder_weights(self):
        """
        对 encoder 中的 Linear 层做 Xavier 初始化，
        保持与原 Encoder._init() 一致。
        """
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        """
        返回：
        - pred: 预测结果
        - feat: encoder 提取到的特征
        """
        feat = self.encoder(x)       # [N, feature_dim]
        pred = self.predictor(feat)  # [N, ...]
        return pred, feat

    def load_model(self, model_path, strict=True):
        """
        可选：从 checkpoint 加载参数。
        如果你后面想用教师 MLP 的权重来初始化学生，可以在外面
        手动做 key 映射；这里先保留一个通用接口。
        """
        checkpoint = torch.load(model_path, map_location="cpu")
        self.load_state_dict(checkpoint, strict=strict)


class SOC_MLPStudentEncoder(nn.Module):
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
        # self.out_ln = nn.LayerNorm(hidden_size)
        self.out_bn = nn.BatchNorm1d(hidden_size, eps=1e-5, momentum=0.1, affine=True)
        self.apply(_init_linear)

        self.fc = nn.Linear(hidden_size, 1)

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
        # h = self.out_ln(self.drop(self.out_proj(pooled)))
        # h = self.drop(self.out_proj(pooled))
        h = self.out_bn(self.drop(self.out_proj(pooled)))
        out = self.fc(h)
        out = out.view(B)
        return out, h, feats

    def forward(self, x: torch.Tensor, lengths=None):
        out, h, _ = self.forward_with_feats(x, lengths)
        return out, h


class SOP_MLPStudentEncoder(nn.Module):
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
        # self.out_ln = nn.LayerNorm(hidden_size)
        self.out_bn = nn.BatchNorm1d(hidden_size, eps=1e-5, momentum=0.1, affine=True)

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
        h = self.out_bn(self.drop(self.out_proj(pooled)))
        return h, feats

    def forward(self, x: torch.Tensor, lengths=None):
        h, _ = self.forward_with_feats(x, lengths)
        return h



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


class FreTS_Model(nn.Module):
    def __init__(self, save_dir='hyperparameters_logs', **kwargs):
        super(FreTS_Model, self).__init__()
        
        # 超参数初始化
        self.embed_size = kwargs.get('embed_size', 64)  # embed_size
        self.hidden_size = kwargs.get('hidden_size', 128)  # hidden_size
        self.pre_length = kwargs.get('pre_length', 64)
        self.feature_size = kwargs.get('feature_size', 3)  # channels
        self.seq_length = kwargs.get('seq_length', 64)
        self.channel_independence = kwargs.get('channel_independence', 0)
        self.sparsity_threshold = kwargs.get('sparsity_threshold', 0.01)
        self.scale = kwargs.get('scale', 0.02)
        self.save_dir = save_dir
        
        # 网络层初始化
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
      
    # dimension extension
    def tokenEmb(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        y = self.embeddings
        return x * y

    def MLP_temporal(self, x, B, N, L):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    def MLP_channel(self, x, B, N, L):
        x = x.permute(0, 2, 1, 3)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        return x

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size], device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size], device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        B, T, N = x.shape
        x = self.tokenEmb(x)
        bias = x
        if self.channel_independence == '1':
            x = self.MLP_channel(x, B, N, T)
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1).reshape(B,-1)
        return x






class SOP_TwoBranchFramework(nn.Module):
    def __init__(self,
                 ):
        super().__init__()
        self.in_norm = nn.InstanceNorm1d(3, affine=False, eps=1e-5)

        self.branch_1 = SOP_TimeSeriesModel(input_dim=3,
                                        conv_channels=[32, 64],
                                        lstm_hidden=128,
                                        lstm_layers=2,
                                        bidirectional=True,
                                        fc_hidden=64,
                                        dropout=0.1,
                                        return_feat=True)

        WINDOW_UDDS = 64
        pre_len = WINDOW_UDDS
        
        hyperparameters = {
            'embed_size': 128,
            'hidden_size': 128,
            'pre_length': pre_len,
            'feature_size': 3,
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
        self.branch_2 = FreTS_Model(**hyperparameters)
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
        xn = raw.transpose(1, 2)
        xn = self.in_norm(xn)
        xn = xn.transpose(1, 2)  # (B, T, 3)

        y1_feat = self.branch_1(xn)

        # 1) 软极小（与“最负电流”一致）
        i_softmin = self.softmin(raw[..., 2], dim=1, keepdim=False)  # [B]

        # 2) soft-abs-argmax：在 |I| 上做 softmax，再加权求 I —— 对应 I_peak 的可导近似
        alpha = 50.0
        i = raw[..., 2]  # [B, T]
        w_abs = torch.softmax(alpha * i.abs(), dim=1)  # [B, T]
        i_softabsmax = (w_abs * i).sum(dim=1)  # [B]

        # 3) SOC 软极小的下降量（逼近标签里的 I_soc 的“分子”，网络自己学比例）
        soc_softmin = self.softmin(raw[..., 1], dim=1, keepdim=False)  # [B]
        soc0 = raw[:, 0, 1]
        soc_drop = soc_softmin - soc0  # [B] <=0

        # 拼成提示向量并投影到与第一支路同维
        hint_vec = torch.stack([i_softmin, i_softabsmax, soc_drop], dim=1)  # [B, 3]
        y1_feat = y1_feat + self.hint_gate * self.hint_proj(hint_vec)

        y2_feat = self.branch_2(xn)
        y2_feat = self.feat2_ln(y2_feat)
        gate = torch.sigmoid(self.branch2_gate)
        y_feat = torch.cat((y1_feat, gate * y2_feat), dim=1)
        y = self.fc(y_feat).view(B)
        return y


class TempConvBlock(nn.Module):
    """简单的 1D temporal conv block: Conv1d -> BN -> ReLU -> Pool（可选）"""
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_chan)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SOP_TimeSeriesModel(nn.Module):
    def __init__(self,
                 input_dim=3,
                 conv_channels=[32, 64],
                 lstm_hidden=128,
                 lstm_layers=2,
                 bidirectional=True,
                 fc_hidden=64,
                 dropout=0.2,
                 return_feat=False):
        super().__init__()
        # Temporal conv: we expect input (batch, seq_len, feat) -> transpose -> (batch, feat, seq_len)
        conv_layers = []
        in_ch = input_dim
        for out_ch in conv_channels:
            conv_layers.append(TempConvBlock(in_ch, out_ch, kernel_size=3))
            in_ch = out_ch
        self.conv_net = nn.Sequential(*conv_layers)  # maps (batch, in_ch, seq_len) -> (batch, out_ch, seq_len)

        self.dropout = nn.Dropout(dropout)
        self.lstm_input_size = conv_channels[-1]
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,  # LSTM expects (batch, seq_len, feature)
                            bidirectional=bidirectional,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.num_directions = 2 if bidirectional else 1
        self.pool_attn = nn.Linear(lstm_hidden * self.num_directions, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * self.num_directions, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)  # predict scalar per sequence
        )
        self.return_feat = return_feat

    def forward(self, x):
        batch = x.size(0)
        seq_len = x.size(1)
        x = x.transpose(1, 2)  # -> (batch, feat, seq_len)
        x = self.conv_net(x)  # -> (batch, conv_channels[-1], seq_len)
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


class RegressorAsEncoder(nn.Module):
    def __init__(self, regressor: nn.Module):
        super().__init__()
        self.backbone = regressor
        self._feat = None

        last_lin = None
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                last_lin = m
        if last_lin is None:
            raise ValueError("[RegressorAsEncoder] 找不到最后的 nn.Linear，无法截取特征")

        self.out_dim = int(last_lin.in_features)

        def _pre_hook(m, inputs):
            x = inputs[0]
            self._feat = x

        self._hook = last_lin.register_forward_pre_hook(_pre_hook)

    def forward(self, x, *args, **kwargs):
        self._feat = None
        _ = self.backbone(x, *args, **kwargs)
        if self._feat is None:
            raise RuntimeError("[RegressorAsEncoder] 未捕获到特征（hook 未触发）")
        f = self._feat
        if f.dim() > 2:
            f = f.reshape(f.size(0), -1)
        return f


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
        self.bnA = nn.BatchNorm1d(dimA, eps=1e-5, momentum=0.1, affine=True)
        self.bnB = nn.BatchNorm1d(dimB, eps=1e-5, momentum=0.1, affine=True)
        self.drop = nn.Dropout(drop)

        self.headA = nn.Linear(dimA, 1, bias=False)
        self.headB = nn.Linear(dimB, 1, bias=False)

    def _star_core(self, x: torch.Tensor):
        B, C, d_series = x.shape

        combined_mean = F.gelu(self.gen1(x))
        combined_mean = self.gen2(combined_mean)

        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1).reshape(-1, C)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(B, -1, 1).permute(0, 2, 1)
            picked = torch.gather(combined_mean, 1, indices)
            combined_mean = picked.repeat(1, C, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True)
            combined_mean = combined_mean.repeat(1, C, 1)

        combined_mean_cat = torch.cat([x, combined_mean], dim=-1)
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
        hA2 = self.bnA(self.drop(self.outA(yA) + hA))  # [B, dimA]
        hB2 = self.bnB(self.drop(self.outB(yB) + hB))  # [B, dimB]

        partA = self.headA(hA2).squeeze(-1)  # [B]
        partB = self.headB(hB2).squeeze(-1)  # [B]
        return partA, partB, hA2, hB2
    


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
    
    A_raw = SOP_TwoBranchFramework()
    B_raw = SOP_TwoBranchFramework()
    
    encA = RegressorAsEncoder(A_raw)
    encB = RegressorAsEncoder(B_raw)
    dimA = encA.out_dim
    dimB = encB.out_dim

    model = WholeNet(encA, encB, dimA, dimB, head_hidden=128, drop=0.10)
    return model


if __name__=='__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # SOH model forward
    print('-'*15 + 'SOH phase' + '-'*15)
    x_soh = torch.randn(64, 13, device=device)
    soh_model = SOH_StudentMLP(input_dim=13, feature_dim=48).to(device)
    y_soh, _ = soh_model(x_soh)
    print(f'x_soh.shape: {x_soh.shape}\ny_soh.shape: {y_soh.shape}')

    # SOC model forward
    print('-'*15 + 'SOC phase' + '-'*15)
    x_soc = torch.randn(512, 64, 3, device=device)
    soc_model = SOC_MLPStudentEncoder(in_dim=3, hidden_size=448).to(device)
    y_soc, _ = soc_model(x_soc)
    print(f'x_soc.shape: {x_soc.shape}\ny_soc.shape: {y_soc.shape}')


    # SOP model forward
    print('-'*15 + 'SOP phase' + '-'*15)
    # 双输入
    xA_sop = torch.randn(512, 64, 3, device=device) 
    xB_sop = torch.randn(512, 64, 3, device=device)
    # 1. 这里创建了SOP Teacher模型，因为SOP Teacher模型中的star模块和head模块还需要推理
    sop_teacher_model = build_whole_model().to(device).eval() 
    # 2. 这里创建了SOP的两个student模型，这两个模型相当于要替换的是teacher模型中的两个特征提取模型
    sop_studentA_model = SOP_MLPStudentEncoder(in_dim=3, hidden_size=64).to(device).eval()
    sop_studentB_model = SOP_MLPStudentEncoder(in_dim=3, hidden_size=64).to(device).eval()
    # 3. 两个student分支各自推理，得到初步的特征图
    sA = sop_studentA_model(xA_sop)
    sB = sop_studentB_model(xB_sop)
    # 4. 使用teacher模型的star模块和head模型完成最后的推理
    _, _, sA2, sB2 = sop_teacher_model.star(sA, sB)
    y_sop = sop_teacher_model.head(torch.cat([sA2, sB2], dim=-1)).squeeze(-1)
    print(f'xA_sop.shape: {xA_sop.shape}\nxB_sop.shape: {xB_sop.shape}\ny_sop.shape: {y_sop.shape}')
