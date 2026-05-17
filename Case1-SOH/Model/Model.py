import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from utils.util import AverageMeter,get_logger,eval_metrix
import os


import snntorch as snn
from snntorch import surrogate
from snntorch import utils
from typing import Optional
from pathlib import Path

from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional

import SeqSNN.module.encoding.snntorch as seqsnn
import SeqSNN.module.encoding.spikingjelly as sj
from SeqSNN.module.spike_attention import Block
from SeqSNN.network.snn.spikformer import Spikformer
from SeqSNN.network.snn.spikegru import TSSNNGRU 


SpikeEncoder = {
    "snntorch": {
        "repeat": seqsnn.encoder.RepeatEncoder,
        "conv": seqsnn.encoder.ConvEncoder,
        "delta": seqsnn.encoder.DeltaEncoder,
    },
    "spikingjelly": {
        "repeat": sj.encoder.RepeatEncoder,
        "conv": sj.encoder.ConvEncoder,
        "delta": sj.encoder.DeltaEncoder,
    },
}


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'




import torch
import torch.nn as nn

class StudentMLP(nn.Module):
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
        super(StudentMLP, self).__init__()

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
        # 和你原来的 MLP 一致： Predictor(input_dim=32)
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

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


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


class Sub_MLP(nn.Module):
    '''
    这个Sub_MLP是作为主干网络的第二分支
    ANN形式
    '''
    def __init__(self,input_dim=17,output_dim=1,layers_num=4,hidden_dim=50,dropout=0.2):
        super(Sub_MLP, self).__init__()

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
        xt = x
        x = xt[:,0:-1]
        t = xt[:,-1:]
        xt = torch.stack((x, t.repeat(1, 12)), dim=-1) # (512, 12, 2)
        x = self.net(xt) # (512, 12, 32)
        b, n, h = x.shape
        x = x.view(b, -1)
        return x


class SpikeLinear(nn.Module):
    '''
    SpikeMLP的子单元
    用来组成SpikeMLP
    '''
    def __init__(self, in_features, out_features, num_steps=4, dropout=0.2):
        super(SpikeLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )
        self.num_steps = num_steps
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """前向传播"""
        x = self.linear(x)
        spk_rec = []
        for _ in range(self.num_steps):
            spk = self.lif.forward(x)
            spk_rec.append(spk)

        spks = torch.stack(spk_rec, dim=-1)
        spks = spks.mean(-1)
        spks = self.dropout(spks)
        return spks


class Spike_MLP(nn.Module):
    '''
    用来作为主干网络的子分支
    '''
    def __init__(self, input_dim=17, output_dim=1, layers_num=4, hidden_dim=50, dropout=0.2, num_steps=16):
        super(Spike_MLP, self).__init__()

        assert layers_num >= 2, "layers must be greater than 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        # 使用 SpikeLinear 代替 nn.Linear
        self.layers = []
        for i in range(layers_num):
            if i == 0:
                # 第一层，使用 SpikeLinear
                self.layers.append(SpikeLinear(input_dim, hidden_dim, num_steps=num_steps, dropout=dropout))
            elif i == layers_num - 1:
                # 最后一层
                self.layers.append(SpikeLinear(hidden_dim, output_dim, num_steps=num_steps, dropout=dropout))
            else:
                # 中间层
                self.layers.append(SpikeLinear(hidden_dim, hidden_dim, num_steps=num_steps, dropout=dropout))

        self.net = nn.Sequential(*self.layers)

    def forward(self, xt):
        x = xt[:, 0:-1]
        t = xt[:, -1:]
        xt_m = torch.stack((x, t.repeat(1, 12)), dim=-1)  # (batch_size, 12, 2)

        # 通过 SpikeLinear 网络
        xt_m = self.net(xt_m)  # (batch_size, 12, hidden_dim)

        # 展平为 (batch_size, -1)
        b, n, h = xt_m.shape
        xt_m = xt_m.view(b, -1)

        return xt_m
    


# no spike_encoder
class SpikeMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_steps=4, dropout=0.2):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.lif1 = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.lif2 = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

        self.dropout = nn.Dropout(dropout)
        # 残差分支，如果输入输出维度不一致
        self.downsample = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
        
        self.lif_res = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        out = self.fc1(x)
        out = self.bn1(out.view(B*S, -1)).view(B, S, -1)
        spk_rec1 = []
        for _ in range(self.num_steps):
            spk = self.lif1.forward(out)
            spk_rec1.append(spk)
        spks1 = torch.stack(spk_rec1, dim=-1).mean(-1)

        out = self.fc2(spks1)
        out = self.bn2(out.view(B*S, -1)).view(B, S, -1)
        out = self.dropout(out)
        spk_rec2 = []
        for _ in range(self.num_steps):
            spk = self.lif2.forward(out)
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


class SpikeingMLP(nn.Module):
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

    def forward(self, xt):
        x = xt[:, 0:-1]
        t = xt[:, -1:]
        xt_m = torch.stack((x, t.repeat(1, 12)), dim=-1)  # (batch_size, 12, 2)
        # xt_m: [B, S, D]  (512, 12, 2)
        out = xt_m
        for block in self.net:
            out = block(out)  # [B, S, hidden_dim]
        # (512, 12, 32)
        b, n, h = out.shape
        out = out.view(b, -1)
        return out  # (512, 384)
    




######################## Spike-iTransformer ########################

tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.d_model = d_model
        self.value_embedding = nn.Linear(c_in, d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, x):
        # x: T B L C
        T, B, _, C = x.shape
        x = x.permute(0, 1, 3, 2).flatten(0, 1)  # TB C L
        x = self.value_embedding(x)  # TB C H
        x = self.bn(x.transpose(-1, -2)).transpose(-1, -2)  # TB C H
        x = x.reshape(T, B, C, self.d_model)
        x = self.lif(x)  # T B C H
        return x


class iSpikformer(nn.Module):
    _snn_backend = "spikingjelly"

    def __init__(
        self,
        dim: int,
        d_ff: Optional[int] = None,
        depths: int = 2,
        common_thr: float = 1.0,
        max_length: int = 100,
        num_steps: int = 4,
        heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = 0.125,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv",
    ):
        super().__init__()
        self.dim = dim
        self.d_ff = d_ff or dim * 4
        self.T = num_steps
        self.depths = depths
        self.encoder = SpikeEncoder[self._snn_backend][encoder_type](num_steps)

        self.emb = DataEmbedding_inverted(max_length, dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    length=max_length,
                    tau=tau,
                    common_thr=common_thr,
                    dim=dim,
                    d_ff=self.d_ff,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                )
                for _ in range(depths)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        functional.reset_net(self.encoder)
        functional.reset_net(self.emb)
        functional.reset_net(self.blocks)
        px = x[:, 0:-1]
        pt = x[:, -1:]
        x = torch.stack((px, pt.repeat(1, 12)), dim=-1)  # (batch_size, 12, 2)
        
        x = self.encoder(x)  # B L C -> T B C L

        x = x.transpose(2, 3)  # T B L C

        x = self.emb(x)  # T B C H
        for blk in self.blocks:
            x = blk(x)  # T B C H
        out = x[-1, :, :, :]
        return out, out  # B C H, B C H

    @property
    def output_size(self):
        return self.dim  # H

    @property
    def hidden_size(self):
        return self.dim


######################## Spike-iTransformer ########################


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


class Solution_u_debug(nn.Module):
    def __init__(self):
        super(Solution_u_debug, self).__init__()
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



class Solution_u(nn.Module):
    def __init__(self):
        super(Solution_u, self).__init__()
        self.encoder = MLP(input_dim=17,output_dim=32,layers_num=3,hidden_dim=60,dropout=0.2)
        self.predictor = Predictor(input_dim=32)
        self._init_()

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)


class Solution_u_v2(nn.Module):
    def __init__(self):
        super(Solution_u_v2, self).__init__()
        self.encoder = MLP(input_dim=13,output_dim=32,layers_num=3,hidden_dim=60,dropout=0.2)
        self.predictor = Predictor(input_dim=32)
        self._init_()

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)


class Solution_u_v3(nn.Module):
    def __init__(self):
        super(Solution_u_v3, self).__init__()
        self.encoder = MLP(input_dim=13,output_dim=32,layers_num=3,hidden_dim=60,dropout=0.2)
        self.spike_encoder= Sub_MLP(input_dim=2,output_dim=32,layers_num=3,hidden_dim=60,dropout=0.2)

        self.predictor = Predictor(input_dim=416)
        self._init_()

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x):
        x1 = self.encoder(x)
        x2 = self.spike_encoder(x)
        x = torch.concat((x1,x2), dim=1)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)


class Solution_u_v4(nn.Module):
    def __init__(self):
        super(Solution_u_v4, self).__init__()
        self.encoder = MLP(input_dim=13,output_dim=32,layers_num=3,hidden_dim=60,dropout=0.2)
        self.spike_encoder= Spike_MLP(input_dim=2,output_dim=32,layers_num=3,hidden_dim=60,dropout=0.2)
        # self.spike_encoder= SpikeingMLP(input_dim=2,seq_len=12, hidden_dim=32,num_layers=3,num_steps=4,dropout=0.2)
        # self.spike_encoder=iSpikformer(dim=16, max_length=12)
        self.predictor = Predictor(input_dim=416)
        self._init_()

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x):
        x1 = self.encoder(x)
        x2 = self.spike_encoder(x)
        x = torch.concat((x1,x2), dim=1)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)


class Solution_u_v5(nn.Module):
    def __init__(self):
        super(Solution_u_v5, self).__init__()
        self.encoder = MLP(input_dim=13,output_dim=32,layers_num=3,hidden_dim=60,dropout=0.2)
        

        ########## v3 ##########
        self.spike_encoder = Spikformer(dim=256, d_ff=1024, 
                                      depths=2, num_steps=4, 
                                      heads=8, pe_type="neuron", 
                                      pe_mode='add', num_pe_neuron=40, 
                                      neuron_pe_scale=10000.0,
                                      input_size=2, max_length=12)
        
        self.spike_proj = nn.Sequential(nn.Linear(256, 32),   # same with MLP out dim
                                        nn.Identity()
                                        )


        # self.spike_encoder = Spikformer(dim=128, d_ff=1024, 
        #                               depths=2, num_steps=4, 
        #                               heads=8, pe_type="neuron", 
        #                               pe_mode='add', num_pe_neuron=40, 
        #                               neuron_pe_scale=10000.0,
        #                               input_size=2, max_length=12)

        # self.spike_encoder = Spikformer(dim=64, d_ff=256, 
        #                               depths=2, num_steps=4, 
        #                               heads=8, pe_type="neuron", 
        #                               pe_mode='add', num_pe_neuron=40, 
        #                               neuron_pe_scale=10000.0,
        #                               input_size=2, max_length=12)

        # self.spike_proj = nn.Sequential(nn.Linear(128, 50),   # same with MLP out dim
        #                                 Sin(),
        #                                 nn.Linear(50, 50),
        #                                 Sin(),
        #                                 nn.Dropout(p=0.2),
        #                                 nn.Linear(50, 50),
        #                                 Sin(),
        #                                 nn.Dropout(p=0.2),
        #                                 nn.Linear(50, 32),
        #                                 # nn.Identity()
        #                                 )
        # self.spike_proj = nn.Sequential(nn.Linear(64, 50),   # same with MLP out dim
        #                                 Sin(),
        #                                 nn.Dropout(p=0.2),
        #                                 nn.Linear(50, 50),
        #                                 Sin(),
        #                                 nn.Linear(50, 32),
        #                                 )


        self.predictor = Predictor(input_dim=416)
        self._init_()

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x):
        x1 = self.encoder(x)

        x_x = x[:, 0:-1]
        x_t = x[:, -1:]
        x2 = torch.stack((x_x, x_t.repeat(1, 12)), dim=-1)  # (batch_size, 12, 2)

        x2, _ = self.spike_encoder(x2) # (batch_size, 12, 128)
        x2 = self.spike_proj(x2) # (batch_size, 12, 32)
        b, _, _ =x2.shape
        x2 = x2.view(b, -1)  # (batch_size, 384)
        x = torch.concat((x1,x2), dim=1) # (batch_size, 416)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)



def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))


class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


class AdaptiveBoundedLoss(nn.Module):
    def __init__(self, lower_bound, upper_bound, alpha=0.1):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.alpha = alpha
        
    def forward(self, predictions):
        
        
        lower_violation = torch.relu(self.lower_bound - predictions)
        upper_violation = torch.relu(predictions - self.upper_bound)
        
        
        violation_degree = (lower_violation + upper_violation).detach()
        adaptive_weights = 1.0 + self.alpha * violation_degree
        
        boundary_penalty = torch.mean(
            adaptive_weights * (lower_violation**2 + upper_violation**2)
        )
        
        return boundary_penalty
    

class PINN_debug(nn.Module):
    '''
    这个版本是在v3的基础上用来加入Spikformer分支继续测试(这个版本的效果感觉还不错)
    还要用于hyx数据
    '''
    def __init__(self,args, debug=False):
        super(PINN_debug, self).__init__()
        self.debug = debug
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.solution_u = Solution_u_debug().to(device)
        self.dynamical_F = MLP(input_dim=27,output_dim=1,  # 35
                               layers_num=args.F_layers_num,
                               hidden_dim=args.F_hidden_dim,
                               dropout=0.2).to(device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.warmup_lr)
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()
        self.bound_func = AdaptiveBoundedLoss(0.0, 1.0)
        # 模型的最好参数(the best model)
        self.best_model = None

        # loss = loss1 + alpha*loss2 + beta*loss3
        self.alpha = self.args.alpha
        self.beta = self.args.beta

    def _save_args(self):
        if self.args.log_dir is not None:
            # 中文： 把parser中的参数保存在self.logger中
            # English: save the parameters in parser to self.logger
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self,xt):
        return self.solution_u(xt)

    def Test(self,testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)

        return true_label,pred_label

    def Valid(self,validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return mse.item()


    def forward(self,xt, return_feat=False):
        xt.requires_grad = True
        x = xt[:,0:-1]
        t = xt[:,-1:]

        if return_feat:
            u, feat = self.solution_u(torch.cat((x,t),dim=1), return_feat)
        else:
            u = self.solution_u(torch.cat((x,t),dim=1), return_feat)


        if return_feat:
            return u, feat

        u_t = grad(u.sum(),t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        u_x = grad(u.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]

        F = self.dynamical_F(torch.cat([xt,u,u_x,u_t],dim=1))

        f = u_t - F
        
        return u,f

    def train_one_epoch(self,epoch,dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0, considering capacity regeneration
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # 加一个边界条件 loss4
            loss4 = 0.5*self.bound_func(u1) + 0.5*self.bound_func(u2) 
            
            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3 + 0.02*loss4

            # W/O Mechanism
            # loss = loss1 + self.beta*loss3 + 0.02*loss4

            # W/O AAOs
            # loss = loss1 + self.alpha*loss2 
            

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()  
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            if self.debug:
                debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
                            "PDE loss:{:.6f}, physics loss:{:.6f}, " \
                            "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())
                self.logger.info(debug_info)
            

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))
        return loss1_meter.avg,loss2_meter.avg,loss3_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            loss1,loss2,loss3 = self.train_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e,current_lr,loss1+self.alpha*loss2+self.beta*loss3)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label,pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u':self.solution_u.state_dict(),
                                   'dynamical_F':self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model,os.path.join(self.args.save_folder,'model.pth'))



class PINN(nn.Module):
    def __init__(self,args):
        super(PINN, self).__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.solution_u = Solution_u().to(device)
        self.dynamical_F = MLP(input_dim=35,output_dim=1,
                               layers_num=args.F_layers_num,
                               hidden_dim=args.F_hidden_dim,
                               dropout=0.2).to(device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.warmup_lr)
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # 模型的最好参数(the best model)
        self.best_model = None

        # loss = loss1 + alpha*loss2 + beta*loss3
        self.alpha = self.args.alpha
        self.beta = self.args.beta

    def _save_args(self):
        if self.args.log_dir is not None:
            # 中文： 把parser中的参数保存在self.logger中
            # English: save the parameters in parser to self.logger
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self,xt):
        return self.solution_u(xt)

    def Test(self,testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)

        return true_label,pred_label

    def Valid(self,validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return mse.item()

    def forward(self,xt):
        xt.requires_grad = True
        x = xt[:,0:-1]
        t = xt[:,-1:]

        u = self.solution_u(torch.cat((x,t),dim=1))

        u_t = grad(u.sum(),t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        u_x = grad(u.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]

        F = self.dynamical_F(torch.cat([xt,u,u_x,u_t],dim=1))

        f = u_t - F
        return u,f

    def train_one_epoch(self,epoch,dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0, considering capacity regeneration
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            # debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
            #              "PDE loss:{:.6f}, physics loss:{:.6f}, " \
            #              "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))
        return loss1_meter.avg,loss2_meter.avg,loss3_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            loss1,loss2,loss3 = self.train_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e,current_lr,loss1+self.alpha*loss2+self.beta*loss3)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label,pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u':self.solution_u.state_dict(),
                                   'dynamical_F':self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model,os.path.join(self.args.save_folder,'model.pth'))


class PINN_v2(nn.Module):
    '''
    这个版本只是用来测试hyx HIS的效果
    '''
    def __init__(self,args):
        super(PINN_v2, self).__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.solution_u = Solution_u_v2().to(device)
        self.dynamical_F = MLP(input_dim=27,output_dim=1,
                               layers_num=args.F_layers_num,
                               hidden_dim=args.F_hidden_dim,
                               dropout=0.2).to(device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.warmup_lr)
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # 模型的最好参数(the best model)
        self.best_model = None

        # loss = loss1 + alpha*loss2 + beta*loss3
        self.alpha = self.args.alpha
        self.beta = self.args.beta

    def _save_args(self):
        if self.args.log_dir is not None:
            # 中文： 把parser中的参数保存在self.logger中
            # English: save the parameters in parser to self.logger
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self,xt):
        return self.solution_u(xt)

    def Test(self,testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)

        return true_label,pred_label

    def Valid(self,validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return mse.item()

    def forward(self,xt):
        xt.requires_grad = True
        x = xt[:,0:-1]
        t = xt[:,-1:]

        u = self.solution_u(torch.cat((x,t),dim=1))

        u_t = grad(u.sum(),t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        u_x = grad(u.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]

        F = self.dynamical_F(torch.cat([xt,u,u_x,u_t],dim=1))

        f = u_t - F
        return u,f

    def train_one_epoch(self,epoch,dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0, considering capacity regeneration
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3
            
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            # debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
            #              "PDE loss:{:.6f}, physics loss:{:.6f}, " \
            #              "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())
            # self.logger.info(debug_info)

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))
        return loss1_meter.avg,loss2_meter.avg,loss3_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            loss1,loss2,loss3 = self.train_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e,current_lr,loss1+self.alpha*loss2+self.beta*loss3)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label,pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u':self.solution_u.state_dict(),
                                   'dynamical_F':self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model,os.path.join(self.args.save_folder,'model.pth'))


class PINN_v3(nn.Module):
    '''
    这个版本是在v2的基础上用来加入SubMLP分支继续测试
    '''
    def __init__(self,args):
        super(PINN_v3, self).__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.solution_u = Solution_u_v3().to(device)
        self.dynamical_F = MLP(input_dim=27,output_dim=1,
                               layers_num=args.F_layers_num,
                               hidden_dim=args.F_hidden_dim,
                               dropout=0.2).to(device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.warmup_lr)
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # 模型的最好参数(the best model)
        self.best_model = None

        # loss = loss1 + alpha*loss2 + beta*loss3
        self.alpha = self.args.alpha
        self.beta = self.args.beta

    def _save_args(self):
        if self.args.log_dir is not None:
            # 中文： 把parser中的参数保存在self.logger中
            # English: save the parameters in parser to self.logger
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self,xt):
        return self.solution_u(xt)

    def Test(self,testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)

        return true_label,pred_label

    def Valid(self,validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return mse.item()

    def forward(self,xt):
        xt.requires_grad = True
        x = xt[:,0:-1]
        t = xt[:,-1:]

        u = self.solution_u(torch.cat((x,t),dim=1))

        u_t = grad(u.sum(),t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        u_x = grad(u.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]

        F = self.dynamical_F(torch.cat([xt,u,u_x,u_t],dim=1))

        f = u_t - F
        return u,f

    def train_one_epoch(self,epoch,dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0, considering capacity regeneration
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
                         "PDE loss:{:.6f}, physics loss:{:.6f}, " \
                         "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())
            self.logger.info(debug_info)

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))
        return loss1_meter.avg,loss2_meter.avg,loss3_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            loss1,loss2,loss3 = self.train_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e,current_lr,loss1+self.alpha*loss2+self.beta*loss3)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label,pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u':self.solution_u.state_dict(),
                                   'dynamical_F':self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model,os.path.join(self.args.save_folder,'model.pth'))


class PINN_v4(nn.Module):
    '''
    这个版本是在v3的基础上用来加入Spike-SubMLP分支继续测试
    '''
    def __init__(self,args):
        super(PINN_v4, self).__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.solution_u = Solution_u_v4().to(device)
        self.dynamical_F = MLP(input_dim=27,output_dim=1,
                               layers_num=args.F_layers_num,
                               hidden_dim=args.F_hidden_dim,
                               dropout=0.2).to(device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.warmup_lr)
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # 模型的最好参数(the best model)
        self.best_model = None

        # loss = loss1 + alpha*loss2 + beta*loss3
        self.alpha = self.args.alpha
        self.beta = self.args.beta

    def _save_args(self):
        if self.args.log_dir is not None:
            # 中文： 把parser中的参数保存在self.logger中
            # English: save the parameters in parser to self.logger
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self,xt):
        return self.solution_u(xt)

    def Test(self,testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)

        return true_label,pred_label

    def Valid(self,validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return mse.item()

    def forward(self,xt):
        xt.requires_grad = True
        x = xt[:,0:-1]
        t = xt[:,-1:]

        u = self.solution_u(torch.cat((x,t),dim=1))

        u_t = grad(u.sum(),t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        u_x = grad(u.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]

        F = self.dynamical_F(torch.cat([xt,u,u_x,u_t],dim=1))

        f = u_t - F
        return u,f

    def train_one_epoch(self,epoch,dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0, considering capacity regeneration
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3
            
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()  # 专门为了spike来的
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            # debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
            #              "PDE loss:{:.6f}, physics loss:{:.6f}, " \
            #              "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))
        return loss1_meter.avg,loss2_meter.avg,loss3_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            loss1,loss2,loss3 = self.train_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e,current_lr,loss1+self.alpha*loss2+self.beta*loss3)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label,pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u':self.solution_u.state_dict(),
                                   'dynamical_F':self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model,os.path.join(self.args.save_folder,'model.pth'))

class PINN_v5(nn.Module):
    '''
    这个版本是在v3的基础上用来加入Spikformer分支继续测试(这个版本的效果感觉还不错)
    '''
    def __init__(self,args):
        super(PINN_v5, self).__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.solution_u = Solution_u_v5().to(device)
        self.dynamical_F = MLP(input_dim=27,output_dim=1,
                               layers_num=args.F_layers_num,
                               hidden_dim=args.F_hidden_dim,
                               dropout=0.2).to(device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.warmup_lr)
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # 模型的最好参数(the best model)
        self.best_model = None

        # loss = loss1 + alpha*loss2 + beta*loss3
        self.alpha = self.args.alpha
        self.beta = self.args.beta

    def _save_args(self):
        if self.args.log_dir is not None:
            # 中文： 把parser中的参数保存在self.logger中
            # English: save the parameters in parser to self.logger
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self,xt):
        return self.solution_u(xt)

    def Test(self,testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)

        return true_label,pred_label

    def Valid(self,validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label),torch.tensor(true_label))
        return mse.item()

    def forward(self,xt):
        xt.requires_grad = True
        x = xt[:,0:-1]
        t = xt[:,-1:]

        u = self.solution_u(torch.cat((x,t),dim=1))

        u_t = grad(u.sum(),t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        u_x = grad(u.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]

        F = self.dynamical_F(torch.cat([xt,u,u_x,u_t],dim=1))

        f = u_t - F
        return u,f

    def train_one_epoch(self,epoch,dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0, considering capacity regeneration
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # 加一个边界条件 loss4

            # 加一个单调项 loss5


            
            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3
            
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()  
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            # debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
            #              "PDE loss:{:.6f}, physics loss:{:.6f}, " \
            #              "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())
            # self.logger.info(debug_info)
            

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))
        return loss1_meter.avg,loss2_meter.avg,loss3_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            loss1,loss2,loss3 = self.train_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e,current_lr,loss1+self.alpha*loss2+self.beta*loss3)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label,pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u':self.solution_u.state_dict(),
                                   'dynamical_F':self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model,os.path.join(self.args.save_folder,'model.pth'))


if __name__ == "__main__":
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch', type=int, default=10, help='1,2,3')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')

        # scheduler 相关
        parser.add_argument('--epochs', type=int, default=1, help='epoch')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epoch')
        parser.add_argument('--warmup_lr', type=float, default=5e-4, help='warmup lr')
        parser.add_argument('--final_lr', type=float, default=1e-4, help='final lr')
        parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of F')
        parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')
        parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
        parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

        parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
        parser.add_argument('--beta', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')

        parser.add_argument('--save_folder', type=str, default=None, help='save folder')
        parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')

        return parser.parse_args()


    args = get_args()
    pinn = PINN(args)
    print(pinn.solution_u)
    count_parameters(pinn.solution_u)
    print(pinn.dynamical_F)




