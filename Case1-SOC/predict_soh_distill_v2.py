import torch
from torch.utils.data import DataLoader
from soh_dataloader import  NASAdata
import argparse
import os
import re
from sklearn import metrics
import numpy as np

import torch.nn as nn
# ============================================================
# 基础配置
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

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






def eval_metrix(true_label,pred_label):
    MAE = metrics.mean_absolute_error(true_label,pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label,pred_label)
    MSE = metrics.mean_squared_error(true_label,pred_label)
    RMSE = np.sqrt(metrics.mean_squared_error(true_label,pred_label))

    return [MAE,MAPE,MSE,RMSE]
# ============================================================
# 模型加载函数
# ============================================================
def load_student_model(model_path: str,
                       input_dim: int = 13,
                       feature_dim: int = 48,
                       map_location=device):
    """
    加载 StudentMLP 模型并返回 model（已经 .eval()）
    """
    model = StudentMLP(input_dim=input_dim, feature_dim=feature_dim).to(device)
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Model] StudentMLP loaded from: {model_path}")
    return model


# ============================================================
# 通用数据加载（原始 UDDS 用）
# ============================================================
def load_UDDS_data(args, normalization=True, small_sample=None):
    root = 'hyx_data/UDDS/csv/'
    train_list = []
    test_list = []

    files = os.listdir(root)
    for file in files:
        if 'C4' in file or 'C14' in file:
            test_list.append(os.path.join(root, file))
        else:
            train_list.append(os.path.join(root, file))

    data = NASAdata(root=root, args=args, normalization=normalization)

    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {
        'train': trainloader['train_2'],
        'valid': trainloader['valid_2'],
        'test': testloader['test_3']
    }

    return dataloader


def load_UDDS_for_predict_data(args, normalization=True):
    root = 'hyx_data/UDDS/csv/'
    test_list = []

    files = os.listdir(root)
    # 按 C 后面的数字排序，例如 HIs_C3_W8.xlsx, HIs_C12_W8.xlsx
    files = sorted(files, key=lambda x: int(re.search(r'C(\d+)_', x).group(1)))

    for file in files:
        if 'C12' in file or 'C13' in file or 'C14' in file:
            test_list.append(os.path.join(root, file))

    data = NASAdata(root=root, args=args, normalization=normalization)

    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'test': testloader['test_4']}

    return dataloader


# ============================================================
# 新增：根据 cid 在数据集路径下寻找文件
# ============================================================
def find_file_by_cid(dataset_root: str,
                     cid: int,
                     exts=('.xlsx', '.xls', '.csv')) -> str:
    """
    在 dataset_root 目录下，根据给定的 cid（例如 12）
    查找包含 'C12' 的文件名，且后缀在 exts 中。
    返回找到的文件的绝对路径。
    若有多个匹配，按文件名字典序取第一个。
    """
    pattern = f"C{cid}"
    files = os.listdir(dataset_root)

    candidate_files = [
        f for f in files
        if pattern in f and f.lower().endswith(tuple(e.lower() for e in exts))
    ]

    if not candidate_files:
        raise FileNotFoundError(
            f"No file containing '{pattern}' with extensions {exts} "
            f"found in {dataset_root}"
        )

    candidate_files.sort()
    chosen = candidate_files[0]
    full_path = os.path.abspath(os.path.join(dataset_root, chosen))
    print(f"[CID] cid={cid}, matched file: {chosen}")
    return full_path


# ============================================================
# 新增：从单个文件构造 DataLoader（整文件为一个 batch）
# ============================================================
def build_dataloader_from_file(args,
                               file_path: str,
                               dataset_root: str = None,
                               normalization: bool = True):
    """
    给定单个文件路径，构造一个 DataLoader：
    - 使用 NASAdata.read_all(specific_path_list=[file_path])
    - 复用 'test_4' 这个 split（如果你实际用的是 'test_3' 就改掉）
    - 用该数据集的长度作为 batch_size（一次性喂给模型）
    """
    file_path = os.path.abspath(file_path)

    if dataset_root is None:
        # 默认认为文件所在目录就是 root
        dataset_root = os.path.dirname(file_path)

    data = NASAdata(root=dataset_root, args=args, normalization=normalization)

    loaders_dict = data.read_all(specific_path_list=[file_path])

    # ⚠️ 这里沿用你 UDDS 预测时用的 'test_4'
    # 如果 NASAdata 那边对应的 key 不同，请改为实际的 key，例如 'test_3'
    base_loader = loaders_dict['test_4']

    dataset = base_loader.dataset
    full_len = len(dataset)

    loader = DataLoader(dataset,
                        batch_size=full_len,
                        shuffle=False,)

    print(f"[Data] Built DataLoader for file: {file_path}, len={full_len}")
    return loader


# ============================================================
# 参数获取
# ============================================================
def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.01, help='base lr')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
    parser.add_argument('--lr_F', type=float, default=0.01, help='lr of F')

    # model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.7, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='adaPINN_test', help='save folder')

    # The AdaPINN class inherits the PINN class, and the above parameters are all parameters of PINN.
    # The following are the parameters of AdaPINN.
    # adaption related
    parser.add_argument('--pretrain_model', type=str, default=None,
                        help='The saving path of the model trained in the source domain')
    parser.add_argument('--adaptation_lr', type=float, default=4e-4, help='adaption lr')
    parser.add_argument('--adaptation_epochs', type=int, default=200, help='adaption epochs')

    parser.add_argument('--target_data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--target_batch', type=int, default=-1, choices=[-1, 0, 1, 2, 3, 4, 5],
                        help='XJTU dataset is divided into 6 batches, and TJU dataset is divided into 3 batches. '
                             'If target_data is XJTU, the value range of target_batch is [-1,0,1,2,3,4,5];'
                             'If target_data is TJU, the value range of target_batch is [-1,0,1,2];'
                             'If it is other datasets, ignore target_batch')

    args = parser.parse_args()

    return args


# ============================================================
# 原有 UDDS 整体预测接口（保留）
# ============================================================
def Predict_UDDS(model='StudentModel'):
    args = get_args()
    predict_task(args, source='UDDS', model=model)


def predict_task(args, source, model='StudentModel'):
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if source in ['UDDS']:
        model_dir = '/workspace/code/nc_proj/soh_pinn_v2/PINN4SOH-main/exp_20251128_distill/UDDS-Ours/Experiment2/student_distill_model.pth'
    else:
        model_dir = f'./pretrained model/model_{source}.pth'

    target_loader = load_UDDS_for_predict_data(args)
    testloader = target_loader['test']

    if model == 'StudentModel':
        student_model = load_student_model(model_dir)
    else:
        raise NotImplementedError(f"Unknown model type: {model}")

    all_true = []
    all_pred = []

    student_model.eval()
    for x1, y1 in testloader:
        x1 = x1.to(device)
        y1 = y1.to(device)

        u1_s, _ = student_model(x1)

        all_true.append(y1.detach().cpu())
        all_pred.append(u1_s.detach().cpu())

    all_true = torch.cat(all_true, dim=0).numpy()
    all_pred = torch.cat(all_pred, dim=0).numpy()

    MAE, MAPE, MSE, RMSE = eval_metrix(all_pred, all_true)
    print(
        "[Distill-Test] MSE: {:.8f}, MAE: {:.6f}, "
        "MAPE: {:.6f}, RMSE: {:.6f}".format(
            MSE, MAE, MAPE, RMSE
        )
    )


# ============================================================
# 新增：给定 cid 与数据集路径，预测 SOH（整文件为一个 batch，输出平均值）
# ============================================================
def predict_soh_for_cid(args,
                        model,
                        cid: int,
                        dataset_root: str,
                        normalization: bool = True) -> float:
    """
    使用给定 model 对指定 cid 对应的文件进行预测：
    - 在 dataset_root 下查找包含 'C{cid}' 的 csv/xlsx 文件
    - 构造 DataLoader（整个文件为一个 batch）
    - 前向推理，收集所有预测
    - 对所有预测值做平均，输出一个 SOH 标量
    """
    # 1) 根据 cid 找到对应文件
    file_path = find_file_by_cid(dataset_root, cid)

    # 2) 构造 DataLoader
    loader = build_dataloader_from_file(args,
                                        file_path=file_path,
                                        dataset_root=dataset_root,
                                        normalization=normalization)

    # 3) 做预测并求平均
    all_pred = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # StudentMLP(x) 返回 (预测值, 特征)
            u1_s, _ = model(x)
            all_pred.append(u1_s.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0).numpy()
    soh_value = float(all_pred.mean())

    print(f"[Predict-SOH] cid={cid}, file={os.path.basename(file_path)}, "
          f"SOH(avg) = {soh_value:.6f}")
    return soh_value


# ============================================================
# 主入口示例
# ============================================================
if __name__ == '__main__':
    args = get_args()

    # 1）加载学生模型
    model_path = '/workspace/code/nc_proj/soh_pinn_v2/PINN4SOH-main/exp_20251128_distill/UDDS-Ours/Experiment2/student_distill_model.pth'
    student_model = load_student_model(model_path)

    # 2）指定数据集路径与 cid，例如 hyx_data/UDDS/csv/ 下的 C12 文件
    dataset_root = 'hyx_data/UDDS/csv/'
    cid = 14

    # 3）对该 cid 对应文件进行预测，整文件为一个 batch，预测结果取平均作为 SOH
    soh = predict_soh_for_cid(args,
                              model=student_model,
                              cid=cid,
                              dataset_root=dataset_root,
                              normalization=True)

    print("Final SOH:", soh)

    # 如果仍然需要原来的 UDDS 整体评估，也可以保留这一行：
    Predict_UDDS('StudentModel')
