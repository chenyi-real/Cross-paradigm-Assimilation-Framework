import math
from typing import Optional
import os
import torch
from torch import nn
import torch.nn.functional as F
import FreTS

def _init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class TCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_size=256, dropout=0.10,
                 use_specmix: bool = True, spec_k: int = 5):
        super().__init__()
        H = hidden_size
        self.hidden_size = H
        self.use_specmix = bool(use_specmix)
        if self.use_specmix:
            self.specmix_head = TM_SpectralMix(d_model=H, k=spec_k, dropout=dropout)

        self.in_conv = nn.Conv1d(in_dim, H, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_uniform_(self.in_conv.weight, a=math.sqrt(5))

        dilations = [1, 2, 4, 8, 16]
        k = 5
        self.dw = nn.ModuleList([
            nn.Conv1d(H, H, kernel_size=k, padding=d*2, dilation=d, groups=H, bias=False)
            for d in dilations
        ])
        self.pw = nn.ModuleList([
            nn.Conv1d(H, H, kernel_size=1, bias=False) for _ in dilations
        ])
        for m in list(self.dw) + list(self.pw):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(H)

        self.attn = nn.Linear(H, 1, bias=False)
        _init_linear(self.attn)

    @torch.no_grad()
    def _make_mask(self, lengths, B, T, device):
        if lengths is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            if lengths.dtype != torch.long:
                lengths = lengths.long()
            t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            mask = (t_idx < lengths.unsqueeze(1))
        row_all_false = ~mask.any(dim=1)
        if row_all_false.any():
            mask[row_all_false, 0] = True
        return mask

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        B, T, D = x.shape
        device = x.device
        mask = self._make_mask(lengths, B, T, device)

        z = x.transpose(1, 2)
        z = self.in_conv(z)

        for dw, pw in zip(self.dw, self.pw):
            y = F.gelu(dw(z))
            y = pw(y)
            z = self.drop(z + y)

        h = z.transpose(1, 2).contiguous()
        if self.use_specmix:
            h = h * mask.unsqueeze(-1).float()
            h = self.specmix_head(h)
        score = self.attn(h).squeeze(-1)
        neg_inf = torch.finfo(score.dtype).min
        score = score.masked_fill(~mask, neg_inf)
        alpha = torch.softmax(score, dim=1)
        ctx = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)

        out = self.ln(self.drop(ctx))
        return out

class TCNRegressor(nn.Module):
    def __init__(self, in_dim: int = 3, hidden_size: int = 64, dropout: float = 0.10):
        super().__init__()
        self.encoder = TCNEncoder(in_dim=in_dim, hidden_size=hidden_size, dropout=dropout)
        self.head = nn.Linear(hidden_size, 1, bias=True)
        _init_linear(self.head)
        self.encoder_output_dim = 64

    def forward(self, x: torch.Tensor, return_feat=False):
        B, T, _ = x.shape
        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        feat = self.encoder(x, lengths)
        y = self.head(feat).squeeze(-1)
        if return_feat:
            return  y, feat
        else:
            return y
    
    def return_encoder_dim(self):
        return  self.encoder_output_dim

    def freeze_layers(self, trainable_layers=("head",)):
        """
        精确控制哪些层可以训练。
        默认仅训练 self.head。
        """
        # 先冻结全部
        for param in self.parameters():
            param.requires_grad = False

        # 精确解冻目标模块
        for layer_name in trainable_layers:
            module = getattr(self, layer_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True
            else:
                print(f"⚠️ 模型中未找到层: {layer_name}")

        # 打印当前可训练参数
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        print(f"🧩 当前可训练参数层: {trainable}")




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
    

    def return_encoder_dim(self):
        return  self.encoder_output_dim

    def freeze_layers(self, trainable_layers=["fc"]):
        """
        冻结模型中除指定层外的所有参数。
        例如：trainable_layers=["fc"] 表示只训练 self.fc。
        支持部分层级，如 ["branch_2", "fc"]。
        """
        # 先冻结所有参数
        for name, param in self.named_parameters():
            param.requires_grad = False

        # 再解冻需要训练的层
        for name, param in self.named_parameters():
            if any(layer in name for layer in trainable_layers):
                param.requires_grad = True

        # 打印结果确认
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        print(f"🧩 可训练参数层：{trainable}")



class MLPModel(nn.Module):
    """
    简单 MLP：直接 flatten 时序特征
    输入: (batch, seq_len, feat)
    输出: (batch,)
    """
    def __init__(self, input_dim=3, seq_len=64, hidden_dims=[64, 64], dropout=0.2, return_feat=False):
        super().__init__()
        in_dim = input_dim * seq_len
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.return_feat = return_feat
        self.encoder_output_dim = 64

    def forward(self, x, return_feat=False):
        batch = x.size(0)
        x = x.view(batch, -1)
        for layer in self.net[:-1]:
            x = layer(x)
        feat = x
        if self.return_feat:
            return feat
        y = self.net[-1](feat).view(batch)
        if return_feat:
            return y, feat
        else:
            return y
    
    def return_encoder_dim(self):
        return  self.encoder_output_dim

    def freeze_layers(self, train_last_only=True):
        """
        冻结所有层，仅保留最后一层可训练。
        如果 train_last_only=False，则解冻所有层。
        """
        if train_last_only:
            # 冻结所有层
            for param in self.parameters():
                param.requires_grad = False
            # 只解冻最后一层线性层
            for param in self.net[-1].parameters():
                param.requires_grad = True
            print("🧊 冻结除最后一层外的所有层，仅训练 self.net[-1]")
        else:
            for param in self.parameters():
                param.requires_grad = True
            print("🔥 解冻所有层，全量训练")

        # 打印当前可训练层名称
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        print(f"🧩 可训练参数层: {trainable}")
    

class CNNModel(nn.Module):
    """
    轻量级 CNN：两层 Conv1d，最大通道 64
    输入: (batch, seq_len, feat)
    输出: (batch,)
    """
    def __init__(self, input_dim=3, conv_channels=[32, 64], kernel_size=3,
                 fc_hidden=64, dropout=0.2, return_feat=False):
        super().__init__()
        layers = []
        in_ch = input_dim
        for out_ch in conv_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.conv_net = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(conv_channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )
        self.return_feat = return_feat

        self.encoder_output_dim = 64

    def forward(self, x, return_feat=False):
        batch = x.size(0)
        x = x.transpose(1, 2)       # (B, feat, seq_len)
        x = self.conv_net(x)        # (B, 64, seq_len/4)
        x = x.mean(dim=2)           # 全局平均池化
        feat = x
        if self.return_feat:
            return feat
        y = self.fc(feat).view(batch)

        if return_feat:
            return y, feat
        else:
            return y
        
    
    def return_encoder_dim(self):
        return  self.encoder_output_dim 
    
    def freeze_layers(self, trainable_layers=["fc"]):
        """
        冻结模型中除指定层外的所有参数。
        例如：trainable_layers=["fc"] 表示只训练 self.fc。
        支持部分层级，如 ["branch_2", "fc"]。
        """
        # 先冻结所有参数
        for name, param in self.named_parameters():
            param.requires_grad = False

        # 再解冻需要训练的层
        for name, param in self.named_parameters():
            if any(layer in name for layer in trainable_layers):
                param.requires_grad = True

        # 打印结果确认
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        print(f"🧩 可训练参数层：{trainable}")



class TwoBranchFramework(nn.Module):
    def __init__(self, exp_dir
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
        
        self.branch_2 = FreTS.Model(exp_dir, **hyperparameters)
        
        self.encoder_output_dim = 256 + 192

        self.fc = nn.Sequential(
            nn.Linear(256 + 192, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, return_feat=False):
        B, T, _ = x.shape
        y1_feat = self.branch_1(x)
        y2_feat = self.branch_2(x)
        y_feat = torch.cat((y1_feat, y2_feat), dim=1)
        y = self.fc(y_feat)
        y = y.view(B)
        if return_feat:
            return y, y_feat
        else:
            return y

    def return_encoder_dim(self):
        return  self.encoder_output_dim 

    def freeze_layers(self, trainable_layers=["fc"]):
        """
        冻结模型中除指定层外的所有参数。
        例如：trainable_layers=["fc"] 表示只训练 self.fc。
        支持部分层级，如 ["branch_2", "fc"]。
        """
        # 先冻结所有参数
        for name, param in self.named_parameters():
            param.requires_grad = False

        # 再解冻需要训练的层
        for name, param in self.named_parameters():
            if any(layer in name for layer in trainable_layers):
                param.requires_grad = True

        # 打印结果确认
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        print(f"🧩 可训练参数层：{trainable}")



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
        # self.out_bn = nn.BatchNorm1d(hidden_size, eps=1e-5, momentum=0.1, affine=True)
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
        h = self.out_ln(self.drop(self.out_proj(pooled)))
        # h = self.drop(self.out_proj(pooled))
        # h = self.out_bn(self.drop(self.out_proj(pooled)))
        out = self.fc(h)
        out = out.view(B)
        return out, h, feats

    def forward(self, x: torch.Tensor, lengths=None):
        out, h, _ = self.forward_with_feats(x, lengths)
        return out, h


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

class TM_SpectralMix(nn.Module):
    def __init__(self, d_model: int, k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv_amp = nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2,
                                  groups=d_model, bias=False)
        self.conv_pha = nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2,
                                  groups=d_model, bias=False)
        self.res_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False)
        self.star = STARFusion(dimA=d_model, dimB=d_model, core_dim=128, drop=dropout)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

        nn.init.kaiming_uniform_(self.conv_amp.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.conv_pha.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.res_conv.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        orig_dtype = x.dtype

        x_res = x
        x_res = x_res.transpose(1, 2)  # [B, D, T]
        x_res = self.res_conv(x_res)
        x_res = x_res.transpose(1, 2)  # [B, T, D]

        X = torch.fft.rfft(x.float(), dim=1)

        amp = X.abs().transpose(1, 2)
        pha = torch.angle(X).transpose(1, 2)

        amp_conv = self.conv_amp(amp)
        pha_conv = self.conv_pha(pha)

        amp_flat = amp_conv.transpose(1, 2).reshape(-1, D)  # [B*F, D]
        pha_flat = pha_conv.transpose(1, 2).reshape(-1, D)  # [B*F, D]

        _, _, amp_fused, pha_fused = self.star(amp_flat, pha_flat)

        amp_hat = amp_fused.view(B, -1, D).transpose(1, 2)  # [B, D, F]
        pha_hat = pha_fused.view(B, -1, D).transpose(1, 2)  # [B, D, F]

        X_hat = amp_hat.transpose(1, 2) * torch.exp(1j * pha_hat.transpose(1, 2))
        y = torch.fft.irfft(X_hat, n=T, dim=1).real

        y = y.to(orig_dtype)
        y = self.drop(self.ln(y + x_res))
        return y

################   SOP    #####################
    
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
        from configs.config_distill_udds import EXP_DIR
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
    
    METHOD = 'Ours'
    EXP_DIR = './TESTETSETEST'
    def _build_teacher_model(device):
        if METHOD == 'Ours':
            model = TwoBranchFramework(exp_dir=EXP_DIR).to(device)
        elif METHOD == 'MLP':
            model = MLPModel(input_dim=3, seq_len=64).to(device)
        elif METHOD == 'TCN':
            model = TCNRegressor(in_dim=3, hidden_size=64, dropout=0.2).to(device)
        elif METHOD == 'LSTM':
            model = TimeSeriesModel(input_dim=3, return_feat=False).to(device)
        elif METHOD == 'CNN':
            model = CNNModel(input_dim=3).to(device)
        else:
            raise ValueError(f"Unknown METHOD: {METHOD}")
        return model
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_teacher_model(device)

    input_tensor = torch.randn((512,64,3)).to(device)
    out = model(input_tensor)