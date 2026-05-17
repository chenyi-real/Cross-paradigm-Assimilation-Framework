import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional
import snntorch as snn
from snntorch import surrogate
from spikingjelly.activation_based import surrogate as sj_surrogate
from pathlib import Path
from snntorch import utils

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



# ====== SOP =======

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

        self.branch_1 = TimeSeriesModel(input_dim=3,
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



# ======= SOH Teacher Model =======
class SOH_Teacher_Model(nn.Module):
    def __init__(self):
        super(SOH_Teacher_Model, self).__init__()
        # self.encoder = MLP(input_dim=13, output_dim=32,layers_num=3,hidden_dim=60,dropout=0.2) 
        self.encoder = CNN()

        self.spike_encoder = TSSNNGRU(num_steps=4, layers=1, hidden_size=32, encoder_type='conv') 

        self.ann_encoder = MLP(input_dim=32,output_dim=16,layers_num=3,hidden_dim=32,dropout=0.2) # 跟SNN在后面

        self.predictor = Predictor(input_dim=48)
        self._init_()

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x, return_feat=False):
        # x: (batch, 13)
        x = x.unsqueeze(1)  # (batch, 1, 13)
        x1 = self.encoder(x) # (batch, 32)

        _, feat_2 = self.spike_encoder(x) # (batch_size, 12, 128)
        # feat = feat_1.mean(dim=-1)
        feat = feat_2
        x2 = self.ann_encoder(feat) # (batch_size, 12, 32)  
        feat = torch.concat((x1,x2), dim=1) # (batch_size, 416)

        x = self.predictor(feat)
        if return_feat:
            return x, feat
        else:
            return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)


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
    

class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),

            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.relu(out)
        return out



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = ResBlock(input_channel=1, output_channel=8, stride=1)  # N,8,17
        self.layer2 = ResBlock(input_channel=8, output_channel=16, stride=2)  # N,16,9
        self.layer3 = ResBlock(input_channel=16, output_channel=24, stride=2)  # N,24,5
        self.layer4 = ResBlock(input_channel=24, output_channel=16, stride=1)  # N,16,5
        self.layer5 = ResBlock(input_channel=16, output_channel=8, stride=1)  # N,8,5
        self.layer6 = nn.Linear(8*4,1)

    def forward(self, x):
        N,L = x.shape[0],x.shape[1]
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out.view(N,-1)



class TSSNNGRU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 1,
        num_steps: int = 50,
        grad_slope: float = 25.0,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv",
    ):
        super().__init__()
        if encoder_type == "conv":
            self.encoder = ConvEncoder(hidden_size)
        elif encoder_type == "delta":
            self.encoder = DeltaEncoder(hidden_size)
        else:
            raise ValueError(f"Unknown encoder type {encoder_type}")

        self.net = nn.Sequential(
            *[
                GRUCell(
                    hidden_size,
                    hidden_size,
                    num_steps=num_steps,
                    grad_slope=grad_slope,
                    output_mems=(i == layers - 1),
                )
                for i in range(layers)
            ]
        )

        self.__output_size = hidden_size

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        for layer in self.net:
            utils.reset(layer)
        hiddens = self.encoder(inputs).squeeze(-1).transpose(1,2)
        _, t, _ = hiddens.size()  # B, L, H
        for i in range(t):
            spks, _ = self.net(hiddens[:, i, :])
        return spks.transpose(1, 2), spks[:, :, -1]  # B * Time Step * H, B * H

    @property
    def output_size(self):
        return self.__output_size


class GRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_steps: int = 4,
        grad_slope: float = 25.0,
        beta: float = 0.99,
        output_mems: bool = False,
    ):
        super().__init__()
        self.spike_grad = surrogate.atan(alpha=2.0)
        self.input_size = input_size
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.beta = beta
        self.full_rec = output_mems
        self.lif = snn.Leaky(
            beta=self.beta,
            spike_grad=self.spike_grad,
            init_hidden=True,
            output=output_mems,
        )
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        self.surrogate_function1 = sj_surrogate.ATan()

    def forward(self, inputs):
        if inputs.size(-1) == self.input_size:
            # assume static spikes:
            h = torch.zeros(
                size=[inputs.shape[0], self.hidden_size],
                dtype=torch.float,
                device=inputs.device,
            )
            y_ih = torch.split(self.linear_ih(inputs), self.hidden_size, dim=1)
            y_hh = torch.split(self.linear_hh(h), self.hidden_size, dim=1)
            r = self.surrogate_function1(y_ih[0] + y_hh[0])
            z = self.surrogate_function1(y_ih[1] + y_hh[1])
            n = self.surrogate_function1(y_ih[2] + r * y_hh[2])
            h = (1.0 - z) * n + z * h
            cur = h
            static = True
        elif inputs.size(-1) == self.num_steps and inputs.size(-2) == self.input_size:
            inputs = inputs.transpose(-1, -2)  # BC, T, H
            h = torch.zeros(
                size=[inputs.shape[0], self.hidden_size, self.num_steps],
                dtype=torch.float,
                device=inputs.device,
            )
            y_ih = torch.split(
                self.linear_ih(inputs).transpose(-1, -2), self.hidden_size, dim=1
            )
            y_hh = torch.split(
                self.linear_hh(h.transpose(-1, -2)).transpose(-1, -2),
                self.hidden_size,
                dim=1,
            )
            r = self.surrogate_function1(y_ih[0] + y_hh[0])
            z = self.surrogate_function1(y_ih[1] + y_hh[1])
            n = self.surrogate_function1(y_ih[2] + r * y_hh[2])
            h = (1.0 - z) * n + z * h
            cur = h
            static = False
        else:
            raise ValueError(
                f"Input size mismatch!"
                f"Got {inputs.size()} but expected (..., {self.input_size}, {self.num_steps}) or (..., {self.input_size})"
            )

        spk_rec = []
        mem_rec = []
        if self.full_rec:
            for i_step in range(self.num_steps):
                if static:
                    spk, mem = self.lif(cur)
                else:
                    spk, mem = self.lif(cur[:, :, i_step])
                spk_rec.append(spk)
                mem_rec.append(mem)
            spks = torch.stack(spk_rec, dim=-1)
            mems = torch.stack(mem_rec, dim=-1)
            return spks, mems
        else:
            for i_step in range(self.num_steps):
                if static:
                    spk = self.lif(cur)
                else:
                    spk = self.lif(cur[:, :, i_step])
                spk_rec.append(spk)
            spks = torch.stack(spk_rec, dim=-1)
            return spks
        

class MLP(nn.Module):
    def __init__(self,input_dim=17,output_dim=1,layers_num=4,hidden_dim=50,dropout=0.2):
        super(MLP, self).__init__()

        assert layers_num >= 2, "layers must be greater than 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim,hidden_dim))
                self.layers.append(Sin())
            elif i == layers_num-1:
                self.layers.append(nn.Linear(hidden_dim,output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                self.layers.append(Sin())
                self.layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*self.layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self,x):
        x = self.net(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.lif = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=False,
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # batch, 1, C, L
        enc = self.encoder(inputs)  # batch, output_size, C, L
        spks = self.lif(enc)
        return spks



class DeltaEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.lif = snn.Leaky(
            beta=0.99, spike_grad=surrogate.atan(), init_hidden=True, output=False
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2)  # batch, 1, C, L
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1)  # batch, C, L, 1
        enc = self.enc(delta)  # batch, C, L, output_size
        enc = enc.permute(0, 3, 1, 2)  # batch, output_size, C, L
        spks = self.lif(enc)
        return spks



# ====== SOC Teacher Model ======
class SOC_Teacher_Model(nn.Module):
    def __init__(self, 
                 ):
        super().__init__()


        self.branch_1 = TimeSeriesModel(input_dim=3,
                            conv_channels=[32, 64],
                            lstm_hidden=128,
                            lstm_layers=2,
                            bidirectional=True,
                            fc_hidden=64,
                            dropout=0.2,
                            return_feat=True)
        
        # 训练脚本示例
        hyperparameters = {
            'embed_size': 64,
            'hidden_size': 128,
            'pre_length': 64,
            'feature_size': 3,
            'seq_length': 64,
            'channel_independence': 1,
            'sparsity_threshold': 0.01,
            'scale': 0.02,
        }
        
        self.branch_2 = FreTS_Model(**hyperparameters)
        
        self.encoder_output_dim = 256 + 192

        self.fc = nn.Sequential(
            nn.Linear(256 + 192, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x, return_feat=False):
        B, T, _ = x.shape
        y1_feat = self.branch_1(x)
        y2_feat = self.branch_2(x)
        y_feat = torch.cat((y1_feat, y2_feat), dim=1)
        y = self.fc(y_feat)
        y = y.view(B, -1, 1)
        if return_feat:
            return y, y_feat
        else:
            return y


# LSTM
class TimeSeriesModel(nn.Module):
    """
    输入: (batch, seq_len, feat) 例如 (128, 64, 3)
    内部: 转为 (batch, feat, seq_len) 用 Conv1d，然后 BiLSTM，再 FC -> 输出 (batch,)
    """
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
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * self.num_directions, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)  # predict scalar per sequence
        )
        self.return_feat = return_feat

        self.encoder_output_dim = 256

    def forward(self, x, return_feat=False):
        """
        x: (batch, seq_len, feat)
        returns: (batch,) tensor
        """
        batch = x.size(0)
        seq_len = x.size(1)
        # conv expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # -> (batch, feat, seq_len)
        x = self.conv_net(x)   # -> (batch, conv_channels[-1], seq_len)
        x = x.transpose(1, 2)  # -> (batch, seq_len, conv_channels[-1]) for LSTM
        # optional dropout on inputs
        x = self.dropout(x)
        # LSTM
        out, (h_n, c_n) = self.lstm(x)  # out: (batch, seq_len, hidden * num_directions)
        # We can use last time-step output (out[:, -1, :]) or pooled; use last step here
        last = out[:, -1, :]  # (batch, hidden * num_directions)
        if self.return_feat:
            return last
        y = self.fc(last)     # (batch, 1)
        y = y.view(batch)     # -> (batch,)
        if return_feat:
            return y, last
        else:
            return y
    


# ====== SOC PINN Model ======

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SOCPINNMModel(nn.Module):
    def __init__(self, d_model, nhead, nhid, nlayers, dropout=0.5, nth_order=8):
        super(SOCPINNMModel, self).__init__()

        self.coder_in = nn.Linear(1,d_model)

        self.conv_in = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=15, padding=15//2), 
            nn.ReLU()
        )

        self.conv_out_1 = nn.Linear(d_model, 1)
        self.conv_out_n = nn.Linear(d_model, nth_order)

        self.coder_pos = PositionalEncoding(d_model, dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=nhid,dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        # 关键的F函数
        self.f_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 关键的H函数
        self.h_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=nhid,dropout=dropout)
        self.h_encoder =  nn.TransformerEncoder(self.h_encoder_layer, num_layers=nlayers)
        
        # 关键的G函数
        self.g_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=nhid,dropout=dropout)
        self.g_encoder = nn.TransformerEncoder(self.g_encoder_layer, num_layers=nlayers)
        

        # 关键的M函数
        self.m_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=nhid,dropout=dropout)
        self.m_encoder = nn.TransformerEncoder(self.m_encoder_layer, num_layers=nlayers)

        # 关键的N函数
        self.n_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=nhid,dropout=dropout)
        self.n_encoder = nn.TransformerEncoder(self.n_encoder_layer, num_layers=nlayers)


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src_soc, src_c, src_mask_soc):
        

        pred_soc = src_soc - src_c * self.f_encoder(torch.cat([src_c, src_soc], dim=2))

        An = self.conv_out_n(self.h_encoder(self.coder_pos(self.conv_in(pred_soc.transpose(2,1)).transpose(2,1)).transpose(0,1), src_mask_soc).transpose(0,1)) # 待修改

        Bn = self.conv_out_n(self.g_encoder(self.coder_pos(self.conv_in(pred_soc.transpose(2,1)).transpose(2,1)).transpose(0,1), src_mask_soc).transpose(0,1)) 


        Vn_sum = (Bn * src_c + An).sum(dim=2, keepdim=True)

        R0 = self.conv_out_1(self.m_encoder(self.coder_pos(self.conv_in(pred_soc.transpose(2,1)).transpose(2,1)).transpose(0,1), src_mask_soc).transpose(0,1))
        
        R0_I = R0 * src_c
        
        Vocv = self.conv_out_1(self.n_encoder(self.coder_pos(self.conv_in(pred_soc.transpose(2,1)).transpose(2,1)).transpose(0,1), src_mask_soc).transpose(0,1))

        pre_v = Vocv - Vn_sum - R0_I

        return pre_v, pred_soc



if __name__=='__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SOH model forward √
    print('-'*15 + 'SOH phase' + '-'*15)
    x_soh = torch.randn(64, 13, device=device)
    soh_model = SOH_Teacher_Model().to(device)
    y_soh = soh_model(x_soh)
    print(f'x_soh.shape: {x_soh.shape}\ny_soh.shape: {y_soh.shape}')


    # SOC model forward
    print('-'*15 + 'SOC phase' + '-'*15)
    x_soc = torch.randn(512, 64, 3, device=device)
    soc_model = SOC_Teacher_Model().to(device)
    y_soc = soc_model(x_soc)
    print(f'x_soc.shape: {x_soc.shape}\ny_soc.shape: {y_soc.shape}')


    # SOC PINN model forward
    print('-'*15 + 'SOC PINN phase' + '-'*15)
    x_I = torch.randn(512, 64, 1, device=device)
    soc_pinn_model = SOCPINNMModel(64, 8, 64, 4, 0.1, 8).to(device)
    src_soc_mask = soc_pinn_model.generate_square_subsequent_mask(y_soc.shape[1]).to(device)
    y_V, y_cal_soc = soc_pinn_model(y_soc, x_I, src_soc_mask)
    print(f'x_I.shape: {x_I.shape}\ny_V.shape: {y_V.shape}\ny_cal_soc.shape: {y_cal_soc.shape}')


    # SOP Teacher model forward √
    print('-'*15 + 'SOP phase' + '-'*15)
    # 双输入
    xA_sop = torch.randn(512, 64, 3, device=device) 
    xB_sop = torch.randn(512, 64, 3, device=device)
    # 1. 这里创建了SOP Teacher模型
    sop_teacher_model = build_whole_model().to(device).eval() 
    y_sop, _ = sop_teacher_model(xA_sop, xB_sop)
    print(f'xA_sop.shape: {xA_sop.shape}\nxB_sop.shape: {xB_sop.shape}\ny_sop.shape: {y_sop.shape}')
