import torch
import torch.nn as nn
from Model.Model import MLP as Encoder
from Model.Model import Predictor
from SeqSNN.network.snn.spikformer import Spikformer as Spikformer_Encoder
from SeqSNN.network.snn.spike_tcn import SpikeTemporalConvNet2D as SpikTCN_Encoder
from SeqSNN.network.snn.spikernn import SpikeRNN as SpikeRNN_Encoder
from SeqSNN.network.snn.spikegru import TSSNNGRU as SpikeGRU_Encoder


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


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.encoder = Encoder(input_dim=13, output_dim=32, layers_num=3, hidden_dim=60, dropout=0.2)
        self.predictor = Predictor(input_dim=32)

    def forward(self,x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x
    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint)

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
        x = x.view(N,1,L)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out.view(N,-1))
        return out.view(N,1)
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint)



class Spikeformer(nn.Module):
    def __init__(self):
        super(Spikeformer, self).__init__()
        self.encoder = Spikformer_Encoder(dim=32, d_ff=128, 
                                      depths=2, num_steps=4, 
                                      heads=8, pe_type="neuron", 
                                      pe_mode='add', num_pe_neuron=40, 
                                      neuron_pe_scale=10000.0,
                                      input_size=13, max_length=500)
        self.predictor = Predictor(input_dim=32)

    def forward(self, x):
        x = x.unsqueeze(1)
        
        _, x = self.encoder(x)
        x = self.predictor(x)
        return x

class SpikeTCN(nn.Module):
    def __init__(self):
        super(SpikeTCN, self).__init__()
        self.encoder = SpikTCN_Encoder(num_levels=3, channel=16, 
                                      dilation=2, stride=1, num_steps=4, 
                                      kernel_size=16, dropout=0., hidden_size=64, 
                                      pe_type="neuron", encoder_type='conv',
                                      pe_mode='add', num_pe_neuron=40, 
                                      neuron_pe_scale=1000.0, input_size=1)
        
        self.predictor = Predictor(input_dim=13)

    def forward(self, x):
        # x = x.unsqueeze(1)
        
        _, x = self.encoder(x)
        x = self.predictor(x)
        return x
    
class SpikeRNN(nn.Module):
    def __init__(self):
        super(SpikeRNN, self).__init__()
        self.encoder = SpikeRNN_Encoder(layers=2, num_steps=4, 
                                      hidden_size=128, 
                                      pe_type="neuron", encoder_type='conv',
                                      pe_mode='add', num_pe_neuron=40, 
                                      neuron_pe_scale=1000.0, input_size=1)
        
        self.predictor = Predictor(input_dim=13)

    def forward(self, x):
        # x = x.unsqueeze(1)
        
        _, x = self.encoder(x)
        x = self.predictor(x)
        return x

class SpikeGRU(nn.Module):
    def __init__(self):
        super(SpikeGRU, self).__init__()
        self.encoder = SpikeGRU_Encoder(num_steps=4, layers=1, hidden_size=32, encoder_type='conv')
        
        self.predictor = Predictor(input_dim=32)

    def forward(self, x):
        x = x.unsqueeze(1)
        
        _, x = self.encoder(x)
        x = self.predictor(x)
        return x
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint)

class LSTM(nn.Module):
    def __init__(self, input_size=13, hidden_size=40, output_size=1, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        N, L = x.shape  # x.shape is (batch_size, sequence_length) => (16, 13)
        
        # LSTM expects (batch_size, sequence_length, input_size), but here input_size is already 13
        x = x.unsqueeze(1)  # Add an extra dimension to make it (batch_size, sequence_length, input_size)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, N, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, N, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Only take the output from the last time step
        out = out[:, -1, :]
        
        # Fully connected layer for output
        out = self.fc(out)
        return out.view(N, 1)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint)


class Chomp1d(nn.Module):
    """裁剪右侧padding，保持因果卷积的输出长度一致"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """TCN中的基础残差模块：因果卷积 + 残差连接"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """完整的TCN网络"""
    def __init__(self, input_size=13, num_channels=[16, 32, 64], kernel_size=3, dropout=0.2, output_size=1):
        super(TCN, self).__init__()
        layers = []
        in_channels = 1  # 输入 (B, 13)，视为单通道序列
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation, padding=padding, dropout=dropout)]
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """
        输入: (B, 13)
        输出: (B, 1)
        """
        N, L = x.shape
        x = x.unsqueeze(1)  # (B, 1, 13)
        out = self.network(x)  # (B, C, 13)
        out = out[:, :, -1]    # 取最后时刻特征
        out = self.fc(out)     # (B, 1)
        return out.view(N, 1)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint)



def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))


if __name__ == '__main__':
    # x = torch.randn(10,17)
    x = torch.randn(10,13)
    y1 = MLP()(x)
    y2 = CNN()(x)
    y3 = Spikeformer()(x)
    count_parameters(CNN())