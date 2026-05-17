import argparse
import random
import time
import json
import pickle
import pandas as pd
import torch
import torch.nn as nn
import os
import numpy as np
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from quality_metrics import *
from model import TimeFreqBranch, MLPRegressor, CNNRegressor, LSTMRegressor, TCNRegressor, timefreq_loss

from torch.utils.data import Dataset
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_tf_input_from_x_raw(x_raw, params, seq_len: int = 16):
    device_local = x_raw.device
    B = x_raw.shape[0]
    T = seq_len

    t_min, t_max = params.time['min'], params.time['max']
    d_min, d_max = params.distance['min'], params.distance['max']

    t = (x_raw[:, 0:1] - t_min) / (t_max - t_min + 1e-8)
    d = (x_raw[:, 1:2] - d_min) / (d_max - d_min + 1e-8)

    td = torch.cat([t, d], dim=1).clamp(0.0, 1.0)   # (B,2)
    td_seq = td.unsqueeze(1).expand(B, T, 2)        # (B,T,2)

    t_idx = torch.linspace(0., 1., T, device=device_local).unsqueeze(0).expand(B, T).unsqueeze(-1)  # (B,T,1)

    x_tf = torch.cat([td_seq, t_idx], dim=-1)       # (B,T,3)
    return x_tf

def get_config(config_path, common_config_path="configs/common.json"):
    with open(common_config_path, 'r') as f:
        config_common = json.load(f)

    with open(config_path, 'r') as f:
        config = json.load(f)

    for key in config_common.keys():
        if key not in config:
            config[key] = config_common[key]

    return ParamsDynamic(config)

def gs_speed_norm(v, v_t, v_d):
    return v_d - 2 * v * v_d - v_t

def get_pde_loss(model, config):
    time_random = torch.from_numpy(
        np.random.uniform(config.time['min'], config.time['max'], (config.n_random_inputs, 1))
    ).float().to(device)
    distance_random = torch.from_numpy(
        np.random.uniform(config.distance['min'], config.distance['max'], (config.n_random_inputs, 1))
    ).float().to(device)

    time_random.requires_grad = True
    distance_random.requires_grad = True
    x_random = torch.cat([time_random, distance_random], dim=-1)

    y_rand_pred = model(x_random)

    u = y_rand_pred
    u_t = torch.autograd.grad(u, time_random, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    u_d = torch.autograd.grad(u, distance_random, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]

    loss_pde = {}
    for pde_model_name in config.physical_model:
        pde_res = globals()[pde_model_name](u, u_t, u_d)
        loss_pde[pde_model_name] = (pde_res ** 2).mean()
    return loss_pde

class NGSIMCustomDataLoader(Dataset):
    def __init__(self, config, df, batch_size=32, mode='train', shuffle=False):
        self.df = df
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.x = torch.from_numpy(df[['time', 'distance']].values).float().to(device)
        self.x_raw = torch.from_numpy(df[['time_raw', 'distance_raw']].values).float().to(device)

        self.y = torch.from_numpy(df[config.mode].values).float().to(device)
        self.y_raw = torch.from_numpy(
            df[[f'{c}_raw' for c in config.mode]].values
        ).float().to(device)

        if self.config.train_sample_method == "random":
            all_idxs = np.arange(len(df))
            np.random.shuffle(all_idxs)
            split_n = int(config.train_sample_p * len(df))
            self.train_idxs = all_idxs[:split_n]
            print(f'Random Data: {int(config.train_sample_p * 100)}% ({split_n} of {len(df)}) samples were assigned for training.')
        else:
            self.train_idxs = np.arange(len(df))

        if mode == 'train':
            self.idxs = self.train_idxs
        else:
            self.idxs = np.arange(len(df))

        self.create_batch()

    def create_batch(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.data = []
        for index in np.array_split(self.idxs, len(self.idxs) // self.batch_size + 1):
            if len(index) == 0:
                continue
            x = self.x[index]
            x_raw = self.x_raw[index]
            y = self.y[index]
            y_raw = self.y_raw[index]
            self.data.append([x, x_raw, y, y_raw])

    def __len__(self):
        return len(self.idxs)

class NN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hs = self.config.nn_hs
        n_layers = self.config.nn_n_layers
        self.bn_hidden = nn.BatchNorm1d(hs)
        self.fc_in = nn.Linear(2, hs)
        self.fc_mids = nn.ModuleList([nn.Linear(hs, hs) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hs, len(config.mode))

        self.init_weights()
        self.ub = torch.Tensor([[float(self.config.time['max']),
                                 float(self.config.distance['max'])]]).to(device)
        self.lb = torch.Tensor([[0., 0.]]).to(device)
        self.isRaw = True

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                m.bias.data.fill_(0.01)

    def forward(self, x):
        if self.isRaw:
            x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        x = self.fc_in(x)
        for l in self.fc_mids:
            if self.config.nn_enable_residual:
                x = torch.tanh(l(x)) + x
            else:
                x = torch.tanh(l(x))
        x = self.fc_out(x)
        return F.relu(x)

class FusionMLP(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int = 32, use_balance: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.use_balance = use_balance
        in_dim = 2 * out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        if self.use_balance:
            self.gate_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, y_pinn, y_tf):
        if self.use_balance:
            gate = torch.sigmoid(self.gate_param)
            z = torch.cat([(1 - gate) * y_pinn, gate * y_tf], dim=-1)
        else:
            z = torch.cat([y_pinn, y_tf], dim=-1)
        y = self.net(z)
        return y

def train_model(args, run_id: int = 0):
    config_path = args.config
    config_name = os.path.basename(config_path).replace('.json', '')
    params = get_config(config_path)

    arch = getattr(args, 'arch', 'pinn_tf').lower()

    base_seed = 114514
    seed = base_seed + run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if args.n_epochs:
        params.dict['n_epochs'] = args.n_epochs

    fusion_hidden = getattr(params, 'fusion_hidden', 32)

    print(f"Training with config: {config_name}")
    print(f"Architecture: {arch}")
    print(f"Epochs: {params.n_epochs}")
    print(f"Fusion MLP hidden dim: {fusion_hidden}")

    df_all = pd.read_csv(f'data/pinn_data_{params.dataset_name}_norm.csv.gz')
    params.dict['time'] = {'max': df_all.time_raw.max(), 'min': 0}
    params.dict['distance'] = {'max': df_all.distance_raw.max(), 'min': 0}
    params.dict['speed'] = {'max': df_all.speed_raw.max(), 'min': 0}

    dataloader = NGSIMCustomDataLoader(
        params, df_all,
        batch_size=params.batch_size,
        shuffle=True
    )

    if arch == "pinn_tf":
        log_root = os.path.join("logs", "pinn_tf", config_name)
    else:
        log_root = os.path.join("logs", arch, config_name)

    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, f"version_{get_next_version(log_root)}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")

    if arch == "pinn_tf":
        pinn_model = NN(params).to(device)
        tf_model = TimeFreqBranch(
            seq_len=16,
            safe_const_norm=(args.pinn_tf_mode in ["tf", "all"])
        ).to(device)
        out_dim = len(params.mode)
        fusion_mlp = None
        if args.pinn_tf_mode == "all":
            fusion_mlp = FusionMLP(
                out_dim = out_dim,
                hidden_dim = fusion_hidden,
                use_balance = True
                ).to(device)
        optimizer_pinn = None
        optimizer_tf = None
        if args.pinn_tf_mode in ("all", "pinn"):
            params_pinn = list(pinn_model.parameters())
            if fusion_mlp is not None:
                params_pinn += list(fusion_mlp.parameters())
            optimizer_pinn = Lamb(params_pinn, lr=params.lr, weight_decay=params.wd)
        if args.pinn_tf_mode in ("all", "tf"):
            optimizer_tf = torch.optim.AdamW(tf_model.parameters(), lr=params.lr, weight_decay=1e-4)

        scheduler_pinn = ReduceLROnPlateau(
            optimizer_pinn if optimizer_pinn is not None else torch.optim.Adam([torch.zeros(1, requires_grad=True)]),
            mode='min',
            patience=params.scheduler_patience,
            factor=0.5,
            min_lr=1e-8
        )
        loss_fn_mse = nn.MSELoss()

        best_loss = float('inf')
        best_pinn_state = None
        best_tf_state = None
        best_fusion_state = None

        loop = tqdm.tqdm(range(1, params.n_epochs + 1), total=params.n_epochs)

        for epoch in loop:
            if args.pinn_tf_mode in ("all", "pinn"):
                pinn_model.train()
            else:
                pinn_model.eval()
            if args.pinn_tf_mode in ("all", "tf"):
                tf_model.train()
            else:
                tf_model.eval()
            if args.pinn_tf_mode == "all" and fusion_mlp is not None:
                fusion_mlp.train()

            total_loss = 0.0
            total_data_loss = 0.0
            total_pde_loss = 0.0
            n_samples = 0

            dataloader.create_batch()

            for x, x_raw, y, y_raw in dataloader.data:
                x_raw = x_raw.to(device)
                y_raw = y_raw.to(device)
                bs = y_raw.size(0)

                if optimizer_pinn is not None:
                    optimizer_pinn.zero_grad(set_to_none=True)
                if optimizer_tf is not None:
                    optimizer_tf.zero_grad(set_to_none=True)

                x_tf = build_tf_input_from_x_raw(
                    x_raw, params, seq_len=16
                )

                y_pinn = None
                y_tf = None
                if args.pinn_tf_mode in ("all", "pinn"):
                    y_pinn = pinn_model(x_raw)
                if args.pinn_tf_mode in ("all", "tf"):
                    y_tf = tf_model(x_tf)
                    if y_tf.dim() == 1:
                        y_tf = y_tf.unsqueeze(-1)
                if args.pinn_tf_mode == "pinn":
                    y_pred = y_pinn
                elif args.pinn_tf_mode == "tf":
                    y_pred = y_tf
                else:
                    if y_pinn.dim() == 1:
                        y_pinn = y_pinn.unsqueeze(-1)
                    y_pred = fusion_mlp(y_pinn, y_tf)
                loss_data = loss_fn_mse(y_raw, y_pred)

                loss_pde_val = 0.0
                if params.loss_pde and args.pinn_tf_mode in ("all", "pinn"):
                    pde_losses = get_pde_loss(pinn_model, params)
                    for v in pde_losses.values():
                        loss_pde_val = loss_pde_val + v

                loss = loss_data + loss_pde_val

                loss.backward()
                if args.pinn_tf_mode in ("all", "pinn"):
                    torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), 1.0)
                if args.pinn_tf_mode == "all" and fusion_mlp is not None:
                    torch.nn.utils.clip_grad_norm_(fusion_mlp.parameters(), 1.0)
                if args.pinn_tf_mode in ("all", "tf"):
                    torch.nn.utils.clip_grad_norm_(tf_model.parameters(), 1.0)

                if optimizer_pinn is not None:
                    optimizer_pinn.step()
                if optimizer_tf is not None:
                    optimizer_tf.step()

                total_loss += float(loss.item()) * bs
                total_data_loss += float(loss_data.item()) * bs
                if isinstance(loss_pde_val, torch.Tensor):
                    total_pde_loss += float(loss_pde_val.item()) * bs
                else:
                    total_pde_loss += float(loss_pde_val) * bs
                n_samples += bs

            avg_loss = total_loss / max(1, n_samples)
            avg_data_loss = total_data_loss / max(1, n_samples)
            avg_pde_loss = total_pde_loss / max(1, n_samples)

            loop.set_postfix({
                "total_loss": avg_loss,
                "data_loss": avg_data_loss,
                "pde_loss": avg_pde_loss
            })

            scheduler_pinn.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                if args.pinn_tf_mode in ("all", "pinn"):
                    best_pinn_state = pinn_model.state_dict()
                    torch.save(best_pinn_state, os.path.join(log_dir, 'checkpoint_pinn_best.pt'))
                if args.pinn_tf_mode in ("all", "tf"):
                    best_tf_state = tf_model.state_dict()
                    torch.save(best_tf_state, os.path.join(log_dir, 'checkpoint_tf_best.pt'))
                if args.pinn_tf_mode == "all" and fusion_mlp is not None:
                    best_fusion_state = fusion_mlp.state_dict()
                    torch.save(best_fusion_state, os.path.join(log_dir, 'checkpoint_fusion_best.pt'))

        print("Training finished (pinn_tf).")

        print("Evaluating model (MLP fused output)...")
        if args.pinn_tf_mode in ("all", "pinn"):
            pinn_model.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoint_pinn_best.pt')))
            pinn_model.eval()
        if args.pinn_tf_mode in ("all", "tf"):
            tf_model.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoint_tf_best.pt')))
            tf_model.eval()
        if args.pinn_tf_mode == "all" and fusion_mlp is not None:
            fusion_mlp.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoint_fusion_best.pt')))
            fusion_mlp.eval()

        df_x_raw = torch.from_numpy(
            df_all[['time_raw', 'distance_raw']].values
        ).float().to(device)

        with torch.no_grad():
            y_pinn = None
            y_tf = None
            if args.pinn_tf_mode in ("all", "pinn"):
                y_pinn = pinn_model(df_x_raw)
            if args.pinn_tf_mode in ("all", "tf"):
                x_tf_all = build_tf_input_from_x_raw(df_x_raw, params, seq_len=16)
                y_tf = tf_model(x_tf_all)
                if y_tf.dim() == 1:
                    y_tf = y_tf.unsqueeze(-1)
            if args.pinn_tf_mode == "all":
                if y_pinn.dim() == 1:
                    y_pinn = y_pinn.unsqueeze(-1)
                y_fused = fusion_mlp(y_pinn, y_tf)

        mode = params.mode[0]
        if y_pinn is not None:
            df_all[f'{mode}_pred_pinn'] = y_pinn[:, 0].detach().cpu().numpy()
        if y_tf is not None:
            df_all[f'{mode}_pred_tf'] = y_tf[:, 0].detach().cpu().numpy()
        if args.pinn_tf_mode == "all":
            df_all[f'{mode}_pred_fused'] = y_fused[:, 0].detach().cpu().numpy()
        if args.pinn_tf_mode == "pinn":
            df_all[f'{mode}_pred'] = df_all[f'{mode}_pred_pinn']
        elif args.pinn_tf_mode == "tf":
            df_all[f'{mode}_pred'] = df_all[f'{mode}_pred_tf']
        else:
            df_all[f'{mode}_pred'] = df_all[f'{mode}_pred_fused']

    else:
        seq_len = getattr(params, "seq_len_tf", 16)
        in_dim = 3

        if arch == "mlp":
            print("[Baseline] Using MLPRegressor (sequence input)")
            model = MLPRegressor(
                input_dim=in_dim,
                seq_len=seq_len,
                hidden_dims=[256, 256],
                dropout=0.10,
                use_in_norm=False
            ).to(device)
        elif arch == "cnn":
            print("[Baseline] Using CNNRegressor")
            model = CNNRegressor(
                in_dim=in_dim,
                channels=(64, 96, 128),
                kernel_size=5,
                dropout=0.10,
                use_in_norm=False
            ).to(device)
        elif arch == "lstm":
            print("[Baseline] Using LSTMRegressor")
            model = LSTMRegressor(
                in_dim=in_dim,
                hidden=128,
                layers=2,
                bidirectional=True,
                dropout=0.10,
                proj_dim=64,
                use_in_norm=False
            ).to(device)
        elif arch == "tcn":
            print("[Baseline] Using TCNRegressor")
            model = TCNRegressor(
                in_dim=in_dim,
                channels=(64, 64, 64, 64),
                kernel_size=3,
                dropout=0.10,
                use_weight_norm=True,
                use_in_norm=False
            ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params.lr, weight_decay=params.wd
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=params.scheduler_patience,
            factor=0.5,
            min_lr=1e-8
        )
        loss_fn_mse = nn.MSELoss()

        best_loss = float("inf")
        best_state = None

        loop = tqdm.tqdm(range(1, params.n_epochs + 1), total=params.n_epochs)
        for epoch in loop:
            model.train()
            dataloader.create_batch()
            epoch_loss = 0.0
            n_samples = 0

            for x, x_raw, y, y_raw in dataloader.data:
                x_tf = build_tf_input_from_x_raw(x_raw, params, seq_len=seq_len).to(device)
                target = y_raw.to(device)
                B = x_tf.size(0)
                n_samples += B

                y_pred = model(x_tf)
                if y_pred.dim() == 1:
                    y_pred = y_pred.unsqueeze(-1)

                loss = loss_fn_mse(y_pred, target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                epoch_loss += loss.item() * B

            epoch_loss /= max(n_samples, 1)
            loop.set_description(f"[{arch}] Epoch {epoch}, loss={epoch_loss:.6f}")

            scheduler.step(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = model.state_dict()
                torch.save(best_state, os.path.join(log_dir, f'checkpoint_{arch}_best.pt'))

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            x_raw_all = torch.from_numpy(
                df_all[['time_raw', 'distance_raw']].values
            ).float().to(device)
            x_tf_all = build_tf_input_from_x_raw(x_raw_all, params, seq_len=seq_len)
            y_pred_all = model(x_tf_all)
            if y_pred_all.dim() == 1:
                y_pred_all = y_pred_all.unsqueeze(-1)

        mode = params.mode[0]
        df_all[f'{mode}_pred_{arch}'] = y_pred_all[:, 0].cpu().numpy()
        df_all[f'{mode}_pred'] = df_all[f'{mode}_pred_{arch}']

    df_all.to_csv(
        os.path.join(log_dir, 'predictions.csv.gz'),
        index=False, compression='gzip'
    )

    results = {}
    for m in params.mode:
        img_real = pd.pivot_table(
            df_all,
            values=f'{m}_raw',
            index='distance_raw',
            columns='time_raw'
        ).values
        img_pred = pd.pivot_table(
            df_all,
            values=f'{m}_pred',
            index='distance_raw',
            columns='time_raw'
        ).values

        img_real = np.expand_dims(img_real, axis=-1)
        img_pred = np.expand_dims(img_pred, axis=-1)

        mse_val = mse(img_real, img_pred)
        mape_val = mape(img_real, img_pred)
        psnr_val = psnr(img_real, img_pred)
        fsim_val = fsim(img_real, img_pred)
        rmse_val = np.sqrt(mse_val)
        mae_val = np.mean(np.abs(img_real - img_pred))

        results[m] = {
            'mse': float(mse_val),
            'rmse': float(rmse_val),
            'mae': float(mae_val),
            'mape': float(mape_val),
            'psnr': float(psnr_val),
            'fsim': float(fsim_val)
        }

    with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4, cls=NpEncoder)

    print("Metrics saved.")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4))
    sns.heatmap(
        pd.pivot_table(
            df_all,
            values=f'{params.mode[0]}_pred',
            index='distance_raw',
            columns='time_raw'
        ),
        cmap="jet_r",
        ax=ax,
        vmin=0,
        vmax=80
    )
    ax.set_title(f"{config_name} ({arch}) | MSE: {results[params.mode[0]]['mse']:.2f}")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.invert_yaxis()
    plt.savefig(os.path.join(log_dir, 'prediction.png'), dpi=300)
    #plt.show()

    print(f"Evaluation finished. Results saved in {log_dir}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PINN + TimeFreq + MLP fused model for traffic flow."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON configuration file.")
    parser.add_argument("--n_epochs", type=int,
                        help="Number of epochs to run, overrides config file.")
    parser.add_argument("--n_runs", type=int, default=20,
                        help="重复训练次数，每次会写到不同的 logs/version_x 目录")
    parser.add_argument(
        "--arch",
        type=str,
        default="pinn_tf",
        choices=["pinn_tf", "mlp", "cnn", "lstm", "tcn"],
        help="选择网络架构：pinn_tf / mlp / cnn / lstm / tcn"
    )
    parser.add_argument(
        "--pinn_tf_mode",
        type=str,
        default="all",
        choices=["all", "pinn", "tf"],
        help="当 arch=pinn_tf 时选择仅用 PINN、仅用 TimeFreq 或二者融合输出"
    )
    args = parser.parse_args()

    # Create common config if it doesn't exist
    if not os.path.exists("configs/common.json"):
        config_common = {
            'mode': ['speed'],
            'physical_model': ['gs_speed_norm'],
            'loss_nn': True,
            'loss_pde': False,
            'train_sample_p': 0.05,
            'main_path': './',
            'random_seed': 42,
            'nn_enable_residual': False,
            'nn_hs': 32,
            'nn_n_layers': 8,
            'n_random_inputs': 4000,
            'n_epochs': 2000,
            'scheduler_patience': 200,
            'batch_size': 4096,
            'optimizer': 'lamb',
            'lr': 4e-3,
            'wd': 2e-4,
            'train_sample_method': 'random',
            'fusion_hidden': 32,
        }
        with open('configs/common.json', 'w', encoding='utf-8') as f:
            json.dump(config_common, f, indent=4)

    for run_idx in range(args.n_runs):
        print(f"\n================ Run {run_idx + 1}/{args.n_runs} ================\n")
        train_model(args, run_id=run_idx)
