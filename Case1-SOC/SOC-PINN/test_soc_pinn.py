
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import nn, Tensor, TupleType

from torch.nn import Transformer



from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

DATASET_CONFIG={'DST-0C-50SOC': 2200,
                'DST-0C-80SOC': 900,
                'DST-25C-50SOC': 3000,
                'DST-25C-80SOC': 2000,
                'DST-45C-50SOC': 2000,
                'DST-45C-80SOC': 2200,

                'FUDS-0C-50SOC': 2000,
                'FUDS-0C-80SOC': 2000,
                'FUDS-25C-50SOC':2500,
                'FUDS-25C-80SOC':2200,
                'FUDS-45C-50SOC':2100,
                'FUDS-45C-80SOC':2000,

                'US06-0C-50SOC': 2200,
                'US06-0C-80SOC': 2000,
                'US06-25C-50SOC':1500,
                'US06-25C-80SOC':1210,
                'US06-45C-50SOC':2100,
                'US06-45C-80SOC':2000,
                }




soc_mm_scaler = MinMaxScaler()
v_mm_scaler = MinMaxScaler()
c_mm_scaler = MinMaxScaler()


# 现在这个数据集的输入是t时刻的SOC，I，t-1时刻的V，输出是t时刻的V
class StrideDataset(Dataset):
    # def __init__(self, file_path, enc_seq_len, target_seq_len, x_size, stride=5):
    def __init__(self, df, enc_seq_len, target_seq_len, stride=5):

        self.src_soc = df.iloc[:, 17:18].values #SoC
        self.src_soc = self.src_soc.squeeze()
        self.src_soc = soc_mm_scaler.fit_transform(self.src_soc.reshape(-1, 1))
        self.src_soc = self.src_soc.squeeze()

        self.src_v = df.iloc[:, 7:8].values #V
        self.src_v = self.src_v.squeeze()
        self.src_v = v_mm_scaler.fit_transform(self.src_v.reshape(-1, 1))
        self.src_v = self.src_v.squeeze()

        self.src_c = df.iloc[:, 6:7].values #I
        self.src_c = self.src_c.squeeze()
        self.src_c = c_mm_scaler.fit_transform(self.src_c.reshape(-1, 1))
        self.src_c = self.src_c.squeeze()


        self.length = len(df)

        num_samples = (self.length - enc_seq_len - target_seq_len) // stride + 1 
        

        src_soc = np.zeros([enc_seq_len, num_samples]) # t-1 时刻

        src_v = np.zeros([enc_seq_len, num_samples])   # t-1 时刻
        
        src_c = np.zeros([enc_seq_len, num_samples])   # t 时刻

        trg_soc = np.zeros([enc_seq_len, num_samples]) # t 时刻
        
        trg_v = np.zeros([enc_seq_len, num_samples])   # t 时刻


        
        for i in np.arange(num_samples):

            # 先计算t-1时刻的
            start_x = stride*i
            end_x = start_x + enc_seq_len
            src_soc[:,i] = self.src_soc[start_x:end_x]
            src_v[:,i] = self.src_v[start_x:end_x]

            # 再计算t时刻的
            start_y = start_x + 5
            end_y = end_x + 5

            src_c[:,i] = self.src_c[start_y:end_y]
            trg_v[:,i] = self.src_v[start_y:end_y]
            trg_soc[:,i] = self.src_soc[start_y:end_y]

        src_soc = src_soc.reshape(src_soc.shape[0], src_soc.shape[1], 1).transpose((1,0,2))

        src_c = src_c.reshape(src_c.shape[0], src_c.shape[1], 1).transpose((1,0,2))

        src_v = src_v.reshape(src_v.shape[0], src_v.shape[1], 1).transpose((1,0,2))
        
        trg_v = trg_v.reshape(trg_v.shape[0], trg_v.shape[1], 1).transpose((1,0,2))

        trg_soc = trg_soc.reshape(trg_soc.shape[0], trg_soc.shape[1], 1).transpose((1,0,2))

        print("src_soc.shape:", src_soc.shape)
        self.src_soc = src_soc
        print("src_c.shape:", src_c.shape)
        self.src_c = src_c
        print("src_v.shape:", src_v.shape)
        self.src_v = src_v
        print("trg_v.shape:", trg_v.shape)
        self.trg_v = trg_v
        print("trg_soc.shape:", trg_soc.shape)
        self.trg_soc = trg_soc

        self.len = len(src_soc)

        self.split = self.len * 0.8

        

    def __getitem__(self, i):
        return self.src_soc[i], self.src_c[i], self.trg_v[i], self.src_v[i], self.trg_soc[i]
    
    def __len__(self):
        return self.len



class CustomDataset(Dataset):
    def __init__(self, df):
        
        #print(df.isnull().sum())

        self.x = df.iloc[:, 6:10].values
        #self.x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
        self.y = df.iloc[:, 7:8].values
        
        self.y_soc = df.iloc[:, 17:18].values
        
        self.length = len(df)

    def __getitem__(self, index):
        # x = torch.FloatTensor([self.x[index]])
        # y = torch.FloatTensor([self.y[index]])
        # return x, y
        feature = torch.FloatTensor([self.x[index+1]])
        label = torch.FloatTensor(self.y[index+1])
        label_soc = torch.FloatTensor(self.y_soc[index+1])
        return feature, label, label_soc

    def __len__(self):
        return self.length




    
class SOCPINNMModel(nn.Module):
    def __init__(self, d_model, nhead, nhid, nlayers, dropout=0.5, nth_order=8):
        super(SOCPINNMModel, self).__init__()

        self.coder_in = nn.Linear(1,d_model)

        self.conv_in = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=15),
            nn.Softmax(dim=1),
            nn.Linear(50-15+1, 50) #src-ker+1, src
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
    
    def forward(self, src_soc, output_soc, src_c, tgt_v, src_mask_soc, src_mask_c, tgt_mask_v):
        
        ####
        # print('main-forward')
        # for item in [src_soc, src_c, tgt_v, src_mask_soc, src_mask_c, tgt_mask_v]:
        #     print(item.shape)
        ####
        
        # 这里写一个函数是预测output_soc的，
        pred_soc = src_soc - src_c * self.f_encoder(torch.cat([src_c, src_soc], dim=2))

        An = self.conv_out_n(self.h_encoder(self.coder_pos(self.conv_in(pred_soc.transpose(2,1)).transpose(2,1)).transpose(0,1), src_mask_soc).transpose(0,1)) # 待修改

        Bn = self.conv_out_n(self.g_encoder(self.coder_pos(self.conv_in(pred_soc.transpose(2,1)).transpose(2,1)).transpose(0,1), src_mask_soc).transpose(0,1)) 


        Vn_sum = (Bn * src_c + An).sum(dim=2, keepdim=True)

        R0 = self.conv_out_1(self.m_encoder(self.coder_pos(self.conv_in(pred_soc.transpose(2,1)).transpose(2,1)).transpose(0,1), src_mask_soc).transpose(0,1))
        
        R0_I = R0 * src_c
        
        Vocv = self.conv_out_1(self.n_encoder(self.coder_pos(self.conv_in(pred_soc.transpose(2,1)).transpose(2,1)).transpose(0,1), src_mask_soc).transpose(0,1))

        pre_v = Vocv - Vn_sum - R0_I

        return pre_v, pred_soc
        


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

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask





def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for MIT dataset')
    parser.add_argument('--data', type=str, default='MIT', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of F')

    # model related
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.02, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='new_results/results of reviewer/NASA results', help='save folder')

    args = parser.parse_args()

    return args



def main(data='USO6', temp='0', level='80', root='./work_dir/'):
 
    dataset_name = f'{data}-{temp}C-{level}SOC'
    workdir = os.path.join(root, data, dataset_name)

    test(workdir=workdir, dataset_name=dataset_name, checkpoint=None)

def test(workdir, dataset_name='FUDS-0C-80SOC', checkpoint=None):
    model = SOCPINNMModel(500, 10, 512, 4, 0.1, 8).to(device)
    
    if checkpoint is None:
        checkpoint = torch.load(f'{workdir}/best_model.pt')
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    df = pd.read_csv(f"./dataset/csv/{dataset_name}.csv")
    df = df[DATASET_CONFIG[dataset_name]:]
    df_len, _ = df.shape
    split = int(df_len * 0.8)
    train_df = df[:split]
    test_df = df[split:]

    eval_dataset = StrideDataset(test_df, 50, 20, stride=1)
    # eval_dataset = StrideDataset("BJDST_80.csv", 50, 20, 1226, stride=1)
    # eval_dataset = StrideDataset("DST_80.csv", 50, 20, 1919, stride=1)
    # eval_dataset = StrideDataset("US06_80.csv", 50, 20, 1207, stride=1)
    eval_dataloader = DataLoader(eval_dataset, batch_size=256, shuffle=False, drop_last=False)

    # 
    dataset_d = CustomDataset(test_df)
    # dataset_d = CustomDataset("BJDST_80.csv", 1226)
    # dataset_d = CustomDataset("DST_80.csv", 1919)
    # dataset_d = CustomDataset("US06_80.csv", 1207)


    predictions_zerosum = torch.zeros(69)
    predictions_zerosum.shape
    predictions_zerosum

    model.eval()

    predictions_v = torch.Tensor(0)
    predictions_soc = torch.Tensor(0)
    # actual = torch.Tensor(0)
    cnt=0

    with torch.no_grad():
        for (inputs_soc, inputs_c, outputs_v, dec_inputs_v, outputs_soc) in eval_dataloader:
            
            # print(inputs_v.shape)
            src_mask_soc = model.generate_square_subsequent_mask(outputs_soc.shape[1]).to(device)
            src_mask_c = model.generate_square_subsequent_mask(inputs_c.shape[1]).to(device)
            # print(model.generate_square_subsequent_mask(inputs_v.shape[1]).shape)

            #print(dec_inputs.shape[1])
            tgt_v_mask = model.generate_square_subsequent_mask(dec_inputs_v.shape[1]).to(device)
            
            pred_v, pred_soc = model(inputs_soc.float().to(device), outputs_soc.float().to(device), inputs_c.float().to(device), dec_inputs_v.float().to(device), src_mask_soc, src_mask_c, tgt_v_mask)

            pred_v = pred_v.permute(1,0,2)
            pred_soc = pred_soc.permute(1,0,2)


            cnt=cnt+1
            predictions_v = torch.cat((predictions_v, pred_v[-1].view(-1).cpu()), 0) # view(-1) => 1

            predictions_soc = torch.cat((predictions_soc, pred_soc[-1].view(-1).cpu()), 0) # view(-1) => 1

            # actual = torch.cat((actual, outputs_v[-1].view(-1).cpu()), 0)


    print(predictions_v.shape)
    print(dataset_d.y.shape)
    print(cnt)

    predictions_v = torch.concat((predictions_zerosum, predictions_v),0)
    predictions_v = v_mm_scaler.inverse_transform(predictions_v.reshape(-1,1))
    predictions_v = predictions_v.squeeze()  #考虑一下是不是需要加这个
    # predictions = predictions * 80.0

    predictions_soc = torch.concat((predictions_zerosum, predictions_soc),0)
    predictions_soc = soc_mm_scaler.inverse_transform(predictions_soc.reshape(-1,1))
    predictions_soc = predictions_soc.squeeze()  #考虑一下是不是需要加这个
    # predictions = predictions * 80.0

    # 画V的图
    plt.figure(figsize=(10,4))
    plt.plot(dataset_d.y[69:], color='red', alpha=0.7)
    plt.plot(predictions_v[69:], color='blue', linewidth=0.7)
    plt.title('Actual vs Forecast')
    plt.legend(['Actual', 'Forecast'])
    plt.xlabel('Time Steps')
    # plt.xlim([2800,3000])
    # plt.ylim([56.5,60.5])
    plt.savefig(os.path.join(workdir, 'V-Actual-vs-Forecast-FUDS.png'))

    df = pd.DataFrame(predictions_v[69:])
    df.to_csv(os.path.join(workdir, '2_F_fuds_v.csv'))

    # 画SOC的图
    plt.figure(figsize=(10,4))
    plt.plot(dataset_d.y_soc[69:1000], color='red', alpha=0.7)
    plt.plot(predictions_soc[69+20:1000], color='blue', linewidth=0.7)
    plt.title('Actual vs Forecast')
    plt.legend(['Actual', 'Forecast'])
    plt.xlabel('Time Steps')
    # plt.xlim([2800,3000])
    # plt.ylim([56.5,60.5])
    plt.savefig(os.path.join(workdir ,'SOC-Actual-vs-Forecast-FUDS.png'))

    df = pd.DataFrame(predictions_soc[69:])
    df.to_csv(os.path.join(workdir ,'2_F_fuds_soc.csv'))


    from sklearn.metrics import mean_absolute_error, mean_squared_error

    print(dataset_d.y.shape)
    print(predictions_v.shape)

    print("v-mae : ", mean_absolute_error(dataset_d.y[69:10000]/ 4.2, predictions_v[69:10000] /4.2))
    print("v-mse : ", mean_squared_error(dataset_d.y[69:10000]/4.2, predictions_v[69:10000]/4.2))
    print("v-rmse : ", np.sqrt(mean_squared_error(dataset_d.y[69:10000]/4.2, predictions_v[69:10000]/4.2)))

    print("soc-mae : ", mean_absolute_error(dataset_d.y_soc[69:10000]/100.0, predictions_soc[69:10000]/100.0))
    print("soc-mse : ", mean_squared_error(dataset_d.y_soc[69:10000]/100.0, predictions_soc[69:10000]/100.0))
    print("soc-rmse : ", np.sqrt(mean_squared_error(dataset_d.y_soc[69:10000]/100.0, predictions_soc[69:10000]/100.0)))



if __name__=='__main__':
    args = get_args()
    index = 1  # 2
    root = './new_work_dir_train/'
    Data = ['DST', 'US06', 'FUDS']
    Temp = ['0', '25', '45']
    Level = ['50', '80']
    if index == 1:
        data_id = 0
        level_id = 0
        for temp in Temp:
            main(data=Data[data_id], temp=temp, level=Level[level_id], root=root)
    elif index == 2:
        data_id = 1
        level_id = 0
        for temp in Temp:
            main(data=Data[data_id], temp=temp, level=Level[level_id], root=root)
    elif index == 3:
        data_id = 2
        level_id = 0
        for temp in Temp:
            main(data=Data[data_id], temp=temp, level=Level[level_id], root=root)
    elif index == 4:
        data_id = 0
        level_id = 1
        for temp in Temp:
            main(data=Data[data_id], temp=temp, level=Level[level_id], root=root)
    elif index == 5:
        data_id = 1
        level_id = 1
        for temp in Temp:
            main(data=Data[data_id], temp=temp, level=Level[level_id], root=root)
    elif index == 6:
        data_id = 2
        level_id = 1
        for temp in Temp:
            main(data=Data[data_id], temp=temp, level=Level[level_id], root=root)
    else:
        print('Error with unknown index!')







