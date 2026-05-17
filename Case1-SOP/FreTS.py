import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import logging

class Model(nn.Module):
    def __init__(self, save_dir='hyperparameters_logs', **kwargs):
        super(Model, self).__init__()
        
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
        # 记录超参数
        self.log_hyperparameters(save_dir)
        
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

    def log_hyperparameters(self, save_dir):
        """记录超参数到文件"""
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 定义超参数字典
        hyperparameters = {
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'pre_length': self.pre_length,
            'feature_size': self.feature_size,
            'seq_length': self.seq_length,
            'channel_independence': self.channel_independence,
            'sparsity_threshold': self.sparsity_threshold,
            'scale': self.scale,
        }
        
        # 记录到 JSON 文件
        log_file = os.path.join(save_dir, 'hyperparameters.json')
        
        # 如果文件存在，先读取文件并追加超参数记录
        if os.path.exists(log_file):
            return
        else:
            data = []
        
        data.append(hyperparameters)
        
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=4)

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
