import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import build_dataloaders_from_udds_df
from configs.config_distill_udds import (
    DEVICE, WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS,
    EPOCHS, LR, PATIENCE, MODELS_DIR, PRETRAIN_WEIGHTS, METHOD, EXP_DIR, DISTILL_WEIGHT
)
from model import TCNRegressor, TimeSeriesModel, TwoBranchFramework, MLPModel, CNNModel, MLPStudentEncoder


def seed_everything(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    if len(y_true) == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    return rmse, mae


def _epoch_distill(model, loader: DataLoader, optimizer=None, device="cpu",
                   desc="", student_model=None, kd_weight=1e-3):
    """
    teacher + student 共同前向的 epoch：
    - teacher 只用于抽特征；
    - student 输出 stu_y 作为最终预测；
    """
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    student_model.train(mode=train_mode)

    total, n = 0.0, 0
    ys, yps, times_all, cids_all = [], [], [], []
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        tb = None
        cids = None
        if isinstance(batch, (list, tuple)):
            if len(batch) == 4:
                xb, yb, tb, cids = batch
            elif len(batch) == 3:
                xb, yb, third = batch
                try:
                    if torch.is_floating_point(third):
                        tb = third
                    else:
                        cids = third
                except Exception:
                    cids = third
            else:
                xb, yb = batch
        else:
            xb, yb = batch

        xb, yb = xb.to(device), yb.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        # student 输出：预测 + 特征
        stu_y, stu_feat = student_model(xb)

        # teacher 输出特征
        _, tea_feat = model(xb, return_feat=True)

        # 监督损失（student -> y）
        stu_y = torch.sigmoid(stu_y)
        sup_loss = F.mse_loss(stu_y, yb)

        # 特征对齐的 KD 损失
        def _l2_normalize(x, eps=1e-6):
            return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps))

        stu_feat_n = _l2_normalize(stu_feat)
        tea_feat_n = _l2_normalize(tea_feat)
        kd_loss = F.mse_loss(stu_feat_n, tea_feat_n)

        loss = sup_loss + kd_weight * kd_loss

        if train_mode:
            loss.backward()
            optimizer.step()

        total += float(loss.item()) * yb.size(0); n += yb.size(0)
        ys.append(yb.detach().cpu().numpy().reshape(-1))
        yps.append(stu_y.detach().cpu().numpy().reshape(-1))
        if tb is not None:
            try:
                times_all.append(tb.detach().cpu().numpy().reshape(-1))
            except Exception:
                times_all.append(np.asarray(tb).reshape(-1))
        if cids is not None:
            try:
                cids_all.append(cids.detach().cpu().numpy().reshape(-1))
            except Exception:
                cids_all.append(np.asarray(cids).reshape(-1))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg = total / max(1, n)
    y  = np.concatenate(ys)  if ys  else np.array([])
    yp = np.concatenate(yps) if yps else np.array([])
    times_flat = np.concatenate(times_all) if times_all else None
    cids_flat  = np.concatenate(cids_all)  if cids_all  else None
    return avg, y, yp, times_flat, cids_flat


def _epoch_eval_teacher(model, loader: DataLoader, device="cpu", desc=""):
    """
    只用 teacher 模型做前向（作为退化备用），不涉及 distill。
    """
    model.eval()
    ys, yps, times_all, cids_all = [], [], [], []
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in pbar:
            tb = None
            cids = None
            if isinstance(batch, (list, tuple)):
                if len(batch) == 4:
                    xb, yb, tb, cids = batch
                elif len(batch) == 3:
                    xb, yb, third = batch
                    try:
                        if torch.is_floating_point(third):
                            tb = third
                        else:
                            cids = third
                    except Exception:
                        cids = third
                else:
                    xb, yb = batch
            else:
                xb, yb = batch

            xb, yb = xb.to(device), yb.to(device)
            # 假定 teacher(x) 直接输出 SOC 预测
            y_pred = model(xb)

            ys.append(yb.detach().cpu().numpy().reshape(-1))
            yps.append(y_pred.detach().cpu().numpy().reshape(-1))
            if tb is not None:
                try:
                    times_all.append(tb.detach().cpu().numpy().reshape(-1))
                except Exception:
                    times_all.append(np.asarray(tb).reshape(-1))
            if cids is not None:
                try:
                    cids_all.append(cids.detach().cpu().numpy().reshape(-1))
                except Exception:
                    cids_all.append(np.asarray(cids).reshape(-1))

    y  = np.concatenate(ys)  if ys  else np.array([])
    yp = np.concatenate(yps) if yps else np.array([])
    times_flat = np.concatenate(times_all) if times_all else None
    cids_flat  = np.concatenate(cids_all)  if cids_all  else None
    return y, yp, times_flat, cids_flat


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


def train_dstill_on_table(train_df, val_df, test_df, device=DEVICE, run_id=1, fname=None):
    """
    训练阶段：teacher + student 蒸馏，并保存两套权重。
    """
    # 构建 DataLoader
    ltr, lva, lte = build_dataloaders_from_udds_df(
        train_df, val_df, test_df, WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS
    )

    # ====== teacher 模型 ======
    model = _build_teacher_model(device)

    # 加载预训练 teacher 权重（可选）
    pretrained_weights = os.path.join(PRETRAIN_WEIGHTS, f"{fname}/best_model_fc_finetuned.pth")
    if pretrained_weights is not None and os.path.exists(pretrained_weights):
        print(f"🔹 加载预训练模型参数：{pretrained_weights}")
        pretrained_state = torch.load(pretrained_weights, map_location=device)
        if isinstance(pretrained_state, dict) and "state_dict" in pretrained_state:
            pretrained_state = pretrained_state["state_dict"]
            model.load_state_dict(pretrained_state.state_dict(), strict=False)
        else:
            model.load_state_dict(pretrained_state, strict=False)
    else:
        print("⚠️ 未检测到 PRETRAIN_WEIGHTS，使用随机初始化权重。")

    # ====== student 模型 ======
    dim_Teacher = model.return_encoder_dim()  # teacher 最后一层特征维度
    model_name = type(model).__name__
    tag = f"[{model_name}|WHOLE]"

    student_model = MLPStudentEncoder(in_dim=3, hidden_size=dim_Teacher).to(device)
    print(f"{tag} [KD] Enable distill: teacher_dim={dim_Teacher}, weight={DISTILL_WEIGHT}")

    base_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    opt_params = base_params + list(student_model.parameters())
    optimizer = torch.optim.AdamW(opt_params, lr=LR, weight_decay=1e-4)

    best_val = float("inf")
    best_teacher_state = None
    best_student_state = None
    bad = 0

    # ====== 训练过程 ======
    for ep in range(1, EPOCHS + 1):
        tr_loss, _, _, _, _ = _epoch_distill(
            model, ltr, optimizer, device,
            desc=f"{tag} Train ep{ep}",
            student_model=student_model, kd_weight=DISTILL_WEIGHT
        )
        va_loss, _, _, _, _ = _epoch_distill(
            model, lva, None, device,
            desc=f"{tag}  Val  ep{ep}",
            student_model=student_model, kd_weight=DISTILL_WEIGHT
        )
        print(f"{tag} ep {ep:02d}: train {tr_loss:.6f} | val {va_loss:.6f}")

        if va_loss < best_val - 1e-12:
            best_val = va_loss
            bad = 0
            best_teacher_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_student_state = {k: v.detach().cpu().clone() for k, v in student_model.state_dict().items()}

            _, y_true, y_pred, times, cids = _epoch_distill(
                model, lte, None, device,
                desc=f"{tag}  Test",
                student_model=student_model, kd_weight=DISTILL_WEIGHT
            )
            rmse, mae = compute_metrics(y_true, y_pred)
            print(f"{tag} ep {ep:02d} Test:  RMSE={rmse:.6f}  MAE={mae:.6f}")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"{tag} Early stop.")
                break

    # ====== 保存模型（teacher + student） ======
    if fname is None:
        root = os.path.join(MODELS_DIR, f"run_{run_id}")
    else:
        root = os.path.join(MODELS_DIR, f"run_{run_id}", os.path.splitext(fname)[0])
    os.makedirs(root, exist_ok=True)

    if best_teacher_state is not None:
        model.load_state_dict(best_teacher_state)
        student_model.load_state_dict(best_student_state)
        teacher_path = os.path.join(root, 'best_model_distill.pth')
        student_path = os.path.join(root, 'best_student_distill.pth')
        torch.save(model.state_dict(), teacher_path)
        torch.save(student_model.state_dict(), student_path)
        print(f"💾 已保存蒸馏后 teacher 模型: {teacher_path}")
        print(f"💾 已保存蒸馏后 student 模型: {student_path}")

    # ====== 用最优权重在 test 上再评估一次（返回） ======
    _, y_true, y_pred, times, cids = _epoch_distill(
        model, lte, None, device,
        desc=f"{tag}  Test (final)",
        student_model=student_model, kd_weight=DISTILL_WEIGHT
    )
    return y_true, y_pred, times, cids


def test_dstill_on_table(test_df, device=DEVICE, run_id=1, fname=None):
    """
    只做测试（给 main.py 的 only_test 分支调用）：
    - 优先加载 teacher + student 的蒸馏权重；
    - 若 student 权重缺失，则退化为仅用 teacher 做预测。
    """
    # 只构建 test dataloader
    _, _, lte = build_dataloaders_from_udds_df(
        test_df, test_df, test_df,
        WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS
    )

    # 构建 teacher / student 结构
    model = _build_teacher_model(device)
    dim_Teacher = model.return_encoder_dim()
    student_model = MLPStudentEncoder(in_dim=3, hidden_size=dim_Teacher).to(device)

    model_name = type(model).__name__
    tag = f"[{model_name}|WHOLE]"

    # 权重路径
    if fname is None:
        root = os.path.join(MODELS_DIR, f"run_{run_id}")
    else:
        root = os.path.join(MODELS_DIR, f"run_{run_id}", os.path.splitext(fname)[0])

    teacher_path = os.path.join(root, 'best_model_distill.pth')
    student_path = os.path.join(root, 'best_student_distill.pth')

    teacher_loaded = False
    student_loaded = False

    if os.path.exists(teacher_path):
        ckpt_t = torch.load(teacher_path, map_location=device)
        model.load_state_dict(ckpt_t)
        teacher_loaded = True
        print(f"🔹 [Test] 加载 teacher 权重: {teacher_path}")
    else:
        print(f"⚠️ [Test] 未找到 teacher 权重: {teacher_path}，将使用随机初始化。")

    if os.path.exists(student_path):
        ckpt_s = torch.load(student_path, map_location=device)
        student_model.load_state_dict(ckpt_s)
        student_loaded = True
        print(f"🔹 [Test] 加载 student 权重: {student_path}")
    else:
        print(f"⚠️ [Test] 未找到 student 权重: {student_path}，将退化为 teacher-only 测试。")

    # ====== 优先使用 student 预测 ======
    if teacher_loaded and student_loaded:
        _, y_true, y_pred, times, cids = _epoch_distill(
            model, lte, None, device,
            desc=f"{tag}  TestOnly(student)",
            student_model=student_model, kd_weight=DISTILL_WEIGHT
        )
        return y_true, y_pred, times, cids
    else:
        # teacher-only 退化模式
        print(f"{tag} [TestOnly] 使用 teacher-only 评估。")
        y_true, y_pred, times, cids = _epoch_eval_teacher(
            model, lte, device=device,
            desc=f"{tag}  TestOnly(teacher)"
        )
        return y_true, y_pred, times, cids


if __name__=="__main__":

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
    