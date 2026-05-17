import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import build_nasa_dataloaders, build_calce_dataloaders, build_mit_dataloaders, build_udds_dataloaders
from config import (
    DEVICE, WINDOW_NASA, STRIDE_NASA, WINDOW_MIT, STRIDE_MIT, BATCH_SIZE, NUM_WORKERS,
    EPOCHS, LR, PATIENCE, NETWORK, WINDOW_CALCE, STRIDE_CALCE, DATASET_NAME,
    UDDS_DIR, WINDOW_UDDS, STRIDE_UDDS, WHOLE_ENABLE, DISTILL_ENABLE, DISTILL_WEIGHT,
    UDDS_GROUP
)
from model import (TwoBranchFramework, TCNRegressor, MLPRegressor,
                   CNNRegressor, LSTMRegressor, build_whole_model, MLPStudentEncoder)

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
            adaptive_weights * (lower_violation ** 2 + upper_violation ** 2)
        )

        return boundary_penalty

class BiasCalibrated(nn.Module):
    def __init__(self, base: nn.Module, init_bias: float = 0.0):
        super().__init__()
        self.base = base
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, x):
        return self.base(x) + self.bias

    def __getattr__(self, name):
        if name in {"base", "bias"}:
            return super().__getattr__(name)
        base = super().__getattr__("base")
        return getattr(base, name)

def _build_model_from_config():
    name = str(NETWORK).lower()
    ds = str(DATASET_NAME).lower()

    if ds == "nasa":
        seq_len = WINDOW_NASA
    elif ds == "mit":
        seq_len = WINDOW_MIT
    elif ds == "calce":
        seq_len = WINDOW_CALCE
    elif ds == "udds":
        seq_len = WINDOW_UDDS
    else:
        seq_len = WINDOW_NASA

    is_calce_voltage = (ds == "calce") and (not WHOLE_ENABLE) and (not DISTILL_ENABLE)
    in_dim = 2 if is_calce_voltage else 3

    if name == "mlp":
        print("[Model] Using MLPRegressor")
        base = MLPRegressor(input_dim=in_dim, seq_len=seq_len, hidden_dims=[256, 256], dropout=0.10)

    elif name == "cnn":
        print("[Model] Using CNNRegressor")
        base = CNNRegressor(in_dim=in_dim, channels=(32, 64, 128), kernel_size=5, dropout=0.10)

    elif name == "lstm":
        print("[Model] Using LSTMRegressor")
        base = LSTMRegressor(in_dim=in_dim, hidden=64, layers=2, bidirectional=True, dropout=0.10)

    elif name == "tcn":
        print("[Model] Using TCNRegressor")
        base = TCNRegressor(in_dim=in_dim, channels=(64, 64, 64, 64), kernel_size=3, dropout=0.10)

    elif name == "two_branch":
        print("[Model] Using TwoBranchFramework")
        base = TwoBranchFramework()

    else:
        raise ValueError(f"未知的网络类型: {NETWORK}，可选：mlp/cnn/lstm/tcn/student/two_branch")

    return base

def _save_truth_pred_plot(times, y_true, y_pred, *, out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    if times is None:
        plt.plot(y_true, label="True")
        plt.plot(y_pred, label="Pred")
    else:
        plt.plot(times, y_true, label="True")
        plt.plot(times, y_pred, label="Pred")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("SOP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    if len(y_true) == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return rmse, mae

def _epoch(model, loader: DataLoader, optimizer=None, device="cpu", desc=""):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
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
                    tb = third if torch.is_floating_point(third) else None
                    cids = None if tb is not None else third
                except Exception:
                    cids = third
            else:
                xb, yb = batch
        else:
            xb, yb = batch

        xb, yb = xb.to(device), yb.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        yhat = model(xb)

        ds = str(DATASET_NAME).lower()
        is_calce_two_branch = (ds == "calce")

        if is_calce_two_branch:
            loss = F.mse_loss(yhat, yb)
        else:
            y_std = yb.std(unbiased=False).clamp_min(5e-4)
            resid = (yhat - yb) / y_std
            data_loss = F.smooth_l1_loss(resid, torch.zeros_like(resid), beta=0.5)

            alpha = 50.0
            i = xb[..., 2]
            soc = xb[..., 1]

            w_abs = torch.softmax(alpha * i.abs(), dim=1)
            i_softabsmax = (w_abs * i).sum(dim=1)
            i_softmin = -torch.logsumexp(-alpha * i, dim=1) / alpha

            soc_softmin = -torch.logsumexp(-alpha * soc, dim=1) / alpha
            soc0 = soc[:, 0]
            i_soc_soft = (soc_softmin - soc0) * model.soc_scale
            target_soft = torch.maximum(i_softabsmax, i_soc_soft)

            aux_max = F.smooth_l1_loss((yhat - target_soft) / y_std, torch.zeros_like(yhat), beta=0.5)
            aux_peak = F.smooth_l1_loss((yhat - i_softabsmax) / y_std, torch.zeros_like(yhat), beta=0.5)
            aux_min = F.smooth_l1_loss((yhat - i_softmin) / y_std, torch.zeros_like(yhat), beta=0.5)

            i_peak_obs = i.amin(dim=1)
            phys_penalty = F.relu(i_peak_obs - yhat).pow(2).mean()

            loss = data_loss + 0.30 * aux_max + 0.10 * aux_peak + 0.05 * aux_min + 0.05 * phys_penalty

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total += float(loss.item()) * yb.size(0)
        n += yb.size(0)
        ys.append(yb.detach().cpu().numpy().reshape(-1))
        yps.append(yhat.detach().cpu().numpy().reshape(-1))
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
    y = np.concatenate(ys) if ys else np.array([])
    yp = np.concatenate(yps) if yps else np.array([])
    times_flat = np.concatenate(times_all) if times_all else None
    cids_flat = np.concatenate(cids_all) if cids_all else None
    return avg, y, yp, times_flat, cids_flat

def _resize_time(x, target_len: int):
    return F.interpolate(x.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)

def train_on_nasa(df, device=DEVICE, save_best_path: str = None):
    ltr, lva, lte = build_nasa_dataloaders(df, WINDOW_NASA, STRIDE_NASA, BATCH_SIZE, NUM_WORKERS)
    model = _build_model_from_config().to(device)
    model_name = type(model).__name__
    tag = f"[{model_name}]"
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=5e-5)

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, EPOCHS + 1):
        tr_loss, _, _, _, _ = _epoch(model, ltr, optimizer, device, desc=f"{tag} Train ep{ep}")
        va_loss, _, _, _, _ = _epoch(model, lva, None, device, desc=f"{tag}  Val  ep{ep}")
        print(f"{tag} ep {ep:02d}: train {tr_loss:.8f} | val {va_loss:.8f} | patience {bad}/{PATIENCE}")
        scheduler.step(va_loss)
        if va_loss < best_val - 1e-12:
            best_val = va_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"{tag} Early stop.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_best_path:
            os.makedirs(os.path.dirname(save_best_path), exist_ok=True)
            torch.save(model.state_dict(), save_best_path)

    _, y_true, y_pred, times, cids = _epoch(model, lte, None, device, desc=f"{tag}  Test")
    return y_true, y_pred, times, cids

def train_on_mit(
        tr_files, va_files, te_files,
        device=DEVICE,
        save_best_path: str = None,
        ckpt_dir: str = None,
        resume: bool = True,
        keep_every_epoch: bool = False,
):
    ltr, lva, lte = build_mit_dataloaders(
        tr_files, va_files, te_files, WINDOW_MIT, STRIDE_MIT, BATCH_SIZE, NUM_WORKERS
    )

    model = _build_model_from_config().to(device)
    model_name = type(model).__name__
    tag = f"[{model_name}]"

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=5e-5
    )

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "last.pt") if ckpt_dir else None
    if save_best_path is None and ckpt_dir:
        save_best_path = os.path.join(ckpt_dir, "best.pt")

    start_ep = 1
    best_val = float("inf")
    bad = 0
    if resume and last_path and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        try:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt and ckpt["scheduler"]:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                except Exception:
                    pass
            start_ep = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", float("inf")))
            bad = int(ckpt.get("bad", 0))
            print(f"{tag} [Resume] epoch={start_ep - 1}, best_val={best_val:.6f}, bad={bad}")
        except Exception as e:
            print(f"{tag} [Resume Failed] {e}，将从头训练。")

    for ep in range(start_ep, EPOCHS + 1):
        tr_loss, _, _, _, _ = _epoch(model, ltr, optimizer, device, desc=f"{tag} Train ep{ep}")
        va_loss, _, _, _, _ = _epoch(model, lva, None, device, desc=f"{tag}  Val  ep{ep}")
        print(f"{tag} ep {ep:02d}: train {tr_loss:.8f} | val {va_loss:.8f} | patience {bad}/{PATIENCE}")
        scheduler.step(va_loss)

        improved = va_loss < best_val - 1e-12
        if improved:
            best_val = va_loss
            bad = 0
            if save_best_path:
                os.makedirs(os.path.dirname(save_best_path), exist_ok=True)
                torch.save(model.state_dict(), save_best_path)
        else:
            bad += 1

        if ckpt_dir:
            state = {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
                "bad": bad,
                "network": type(model).__name__,
                "distill": bool(DISTILL_ENABLE),
            }
            torch.save(state, last_path)
            if keep_every_epoch:
                torch.save(state, os.path.join(ckpt_dir, f"epoch_{ep:03d}.pt"))

        if bad >= PATIENCE:
            print(f"{tag} Early stop.")
            break

    if save_best_path and os.path.isfile(save_best_path):
        obj = torch.load(save_best_path, map_location=device)
        if isinstance(obj, dict) and "model" in obj:
            model.load_state_dict(obj["model"])
        else:
            model.load_state_dict(obj)

    _, y_true, y_pred, times, cids = _epoch(model, lte, None, device, desc=f"{tag}  Test")
    return y_true, y_pred, times, cids

def train_on_calce(tr_files, va_files, te_files, device=DEVICE, ckpt_dir: str = None,
                   save_best_path: str = None, resume: bool = True, keep_every_epoch: bool = False):
    ltr, lva, lte = build_calce_dataloaders(
        tr_files, va_files, te_files, WINDOW_CALCE, STRIDE_CALCE, BATCH_SIZE, NUM_WORKERS
    )
    model = _build_model_from_config().to(device)

    if hasattr(model, "base"):
        tag = f"[{type(model.base).__name__}]"
    else:
        tag = f"[{type(model).__name__}]"

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=5e-5)

    last_path = None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        last_path = os.path.join(ckpt_dir, "last.pt")

        if save_best_path is None:
            save_best_path = os.path.join(ckpt_dir, "best.pt")

    start_ep = 1
    best_val = float("inf")
    bad = 0
    if resume and last_path and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        try:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])

            if "scheduler" in ckpt and ckpt["scheduler"]:

                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                except Exception:
                    pass
            start_ep = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", float("inf")))
            bad = int(ckpt.get("bad", 0))
            print(f"{tag} [Resume] epoch={start_ep - 1}, best_val={best_val:.6f}, bad={bad}")
        except Exception as e:
            print(f"{tag} [Resume Failed] {e}，将从头训练。")

    best_state = None
    for ep in range(start_ep, EPOCHS + 1):
        tr_loss, _, _, _, _ = _epoch(model, ltr, optimizer, device, desc=f"{tag} Train ep{ep}")
        va_loss, _, _, _, _ = _epoch(model, lva, None, device, desc=f"{tag}  Val  ep{ep}")
        print(f"{tag} ep {ep:02d}: train {tr_loss:.8f} | val {va_loss:.8f} | patience {bad}/{PATIENCE}")
        scheduler.step(va_loss)
        improved = va_loss < best_val - 1e-12
        if improved:
            best_val, bad = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_best_path:
                os.makedirs(os.path.dirname(save_best_path), exist_ok=True)
                torch.save(model.state_dict(), save_best_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"{tag} Early stop.")
                break

        if ckpt_dir and last_path:
            state = {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
                "bad": bad,
                "network": type(model).__name__,
            }
            torch.save(state, last_path)
            if keep_every_epoch:
                torch.save(state, os.path.join(ckpt_dir, f"epoch_{ep:03d}.pt"))

    if best_state is not None:
        model.load_state_dict(best_state)
    elif save_best_path and os.path.isfile(save_best_path):
        model.load_state_dict(torch.load(save_best_path, map_location=device))

    _, y_true, y_pred, times, cids = _epoch(model, lte, None, device, desc=f"{tag}  Test")
    return y_true, y_pred, times, cids

def _epoch_whole(model, loader: DataLoader, optimizer=None, device=DEVICE, desc="", bound_weight=0.0,
                 bound_loss_fn=None):
    training = optimizer is not None
    model.train(training)
    tot, n = 0.0, 0
    ys, yps, times_all, cids_all = [], [], [], []

    for xA, xB, y, t_end, *rest in tqdm(loader, desc=desc, leave=False):
        xA = xA.to(device)
        xB = xB.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        yhat, _ = model(xA, xB)

        y_std = y.detach().std(unbiased=False).clamp_min(5e-2)
        resid = (yhat - y) / y_std
        main_loss = F.smooth_l1_loss(resid, torch.zeros_like(resid), beta=0.5)

        loss = main_loss
        if bound_loss_fn is not None and training:
            b_loss = bound_loss_fn(yhat)
            loss = loss + bound_weight * b_loss

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = y.size(0)
        tot += float(loss.item()) * bs
        n += bs

        ys.append(y.detach().cpu().numpy().reshape(-1))
        yps.append(yhat.detach().cpu().numpy().reshape(-1))

        try:
            times_all.append(t_end.detach().cpu().numpy().reshape(-1))
        except Exception:
            import numpy as _np
            times_all.append(_np.asarray(t_end).reshape(-1))

        if rest:
            cids = rest[0]
            try:
                cids_all.append(cids.detach().cpu().numpy().reshape(-1))
            except Exception:
                import numpy as _np
                cids_all.append(_np.asarray(cids).reshape(-1))

    y = np.concatenate(ys) if ys else np.array([])
    yp = np.concatenate(yps) if yps else np.array([])
    t = np.concatenate(times_all) if times_all else None
    cid = np.concatenate(cids_all) if cids_all else None
    return tot / max(1, n), y, yp, t, cid

def _teacher_best_path():
    return os.path.join(
        ".", "checkpoints_whole", "two_branch", str(UDDS_GROUP), "run_1", "best.pt"
    )

def _freeze_module(m: torch.nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

def _epoch_whole_distill(model, loader: DataLoader, optimizer=None, device=DEVICE, desc="",
                         studentA=None, studentB=None, student_star=None, student_head=None,
                         kd_out: float = 0.0, kd_feat: float = 0.0, enable_feat_kd: bool = True):
    printed_kd = False
    training = optimizer is not None
    model.eval()
    if studentA is not None:
        studentA.train(training)
    if studentB is not None:
        studentB.train(training)

    tot, n = 0.0, 0
    ys, yps, times_all, cids_all = [], [], [], []

    for xA, xB, y, t_end, *rest in tqdm(loader, desc=desc, leave=False):
        xA = xA.to(device)
        xB = xB.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        sA = studentA(xA)
        sB = studentB(xB)
        _, _, sA2, sB2 = student_star(sA, sB)
        y_student = student_head(torch.cat([sA2, sB2], dim=-1)).squeeze(-1)

        y_std = y.detach().std(unbiased=False).clamp_min(5e-2)
        resid = (y_student - y) / y_std
        sup_loss = F.smooth_l1_loss(resid, torch.zeros_like(resid), beta=0.5)

        with torch.no_grad():
            y_teacher, (tA2, tB2) = model(xA, xB)

        def _l2_normalize(x, eps=1e-6):
            return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps))

        tA_n = _l2_normalize(tA2)
        tB_n = _l2_normalize(tB2)
        sA_n = _l2_normalize(sA2)
        sB_n = _l2_normalize(sB2)

        kdA = F.smooth_l1_loss(sA_n, tA_n, beta=0.5)
        kdB = F.smooth_l1_loss(sB_n, tB_n, beta=0.5)

        y_teacher = y_teacher.view_as(y_student)

        tsout_loss = F.smooth_l1_loss(y_teacher, y_student, beta=0.5)

        feat_term = (kdA + kdB) if enable_feat_kd else torch.zeros_like(kdA)

        loss = sup_loss + kd_feat * feat_term + kd_out * tsout_loss

        if training and not printed_kd:
            print(f"[KD Debug] sup={sup_loss.detach().item():.4f} "
                  f"kdA={kdA.detach().item():.4f} "
                  f"kdB={kdB.detach().item():.4f}")
            printed_kd = True

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(studentA.parameters()) + list(studentB.parameters()),
                1.0
            )
            optimizer.step()

        bs = y.size(0)
        tot += float(loss.item()) * bs
        n += bs

        ys.append(y.detach().cpu().numpy().reshape(-1))
        yps.append(y_student.detach().cpu().numpy().reshape(-1))

        try:
            times_all.append(t_end.detach().cpu().numpy().reshape(-1))
        except Exception:
            import numpy as _np
            times_all.append(_np.asarray(t_end).reshape(-1))

        if rest:
            cids = rest[0]
            try:
                cids_all.append(cids.detach().cpu().numpy().reshape(-1))
            except Exception:
                import numpy as _np
                cids_all.append(_np.asarray(cids).reshape(-1))

    y = np.concatenate(ys) if ys else np.array([])
    yp = np.concatenate(yps) if yps else np.array([])
    t = np.concatenate(times_all) if times_all else None
    cid = np.concatenate(cids_all) if cids_all else None
    return tot / max(1, n), y, yp, t, cid

def train_whole_on_udds(device=DEVICE,
                        save_best_path: str = None,
                        ckpt_dir: str = None,
                        resume: bool = True,
                        keep_every_epoch: bool = False,
                        loaders=None):
    if loaders is None:
        ltr, lva, lte = build_udds_dataloaders(
            UDDS_DIR, WINDOW_UDDS, STRIDE_UDDS,
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
        )
    else:
        ltr, lva, lte = loaders

    model = build_whole_model().to(device)
    model_name = type(model).__name__
    tag = f"[{model_name}|WHOLE]"

    if DISTILL_ENABLE:
        teacher_path = _teacher_best_path()
        if not os.path.isfile(teacher_path):
            raise FileNotFoundError(f"[Teacher] best.pt not found: {teacher_path}")

        ckpt = torch.load(teacher_path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        _freeze_module(model)
        print(f"{tag} [Teacher] Loaded & frozen: {teacher_path}")

    studentA = studentB = None
    student_star = None
    student_head = None
    if DISTILL_ENABLE:
        dimA = int(model.encoder_a.out_dim)
        dimB = int(model.encoder_b.out_dim)
        studentA = MLPStudentEncoder(in_dim=3, hidden_size=dimA).to(device)
        studentB = MLPStudentEncoder(in_dim=3, hidden_size=dimB).to(device)
        student_star = model.star
        student_head = model.head
        _freeze_module(student_star)
        _freeze_module(student_head)
        print(f"{tag} [KD] Enable distill: A_dim={dimA}, B_dim={dimB}, weight={DISTILL_WEIGHT}")

    bound_loss_fn = None
    BOUND_WEIGHT = 0.1

    if not DISTILL_ENABLE:
        print(f"{tag} [Auto-Bounds] Scanning training data for min/max limits...")
        y_min_val, y_max_val = float("inf"), float("-inf")

        for batch in tqdm(ltr, desc="[Scanning Bounds]", leave=False):
            y_batch = batch[2]

            current_min = y_batch.min().item()
            current_max = y_batch.max().item()

            if current_min < y_min_val: y_min_val = current_min
            if current_max > y_max_val: y_max_val = current_max

        print(f"{tag} [Auto-Bounds] Detected Range: [{y_min_val:.4f}, {y_max_val:.4f}]")

        bound_loss_fn = AdaptiveBoundedLoss(y_min_val, y_max_val, alpha=0.1).to(device)

    if DISTILL_ENABLE:
        opt_params = list(studentA.parameters()) + list(studentB.parameters())
    else:
        opt_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(opt_params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=5e-5
    )

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "last.pt") if ckpt_dir else None
    if save_best_path is None and ckpt_dir:
        save_best_path = os.path.join(ckpt_dir, "best.pt")

    start_ep = 1
    best_val = float("inf")
    bad = 0
    if resume and last_path and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        try:
            if not DISTILL_ENABLE:
                model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt and ckpt["scheduler"]:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                except Exception:
                    pass
            if DISTILL_ENABLE and ckpt.get("distill", False):
                if "studentA" in ckpt and studentA is not None:
                    studentA.load_state_dict(ckpt["studentA"])
                if "studentB" in ckpt and studentB is not None:
                    studentB.load_state_dict(ckpt["studentB"])

            start_ep = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", float("inf")))
            bad = int(ckpt.get("bad", 0))
            print(f"{tag} [Resume] epoch={start_ep - 1}, best_val={best_val:.6f}, bad={bad}")
        except Exception as e:
            print(f"{tag} [Resume Failed] {e}，将从头训练。")

    if not DISTILL_ENABLE:
        max_epochs = EPOCHS
        print(f"{tag} [Epochs] DISTILL disabled & arch=two_branch, use EPOCHS={max_epochs}.")
    else:
        max_epochs = EPOCHS

    KD_WARMUP_EPOCHS = max(3, EPOCHS // 5)
    KD_FEAT_RATIO = 0.03
    ENABLE_FEAT_KD = True

    for ep in range(start_ep, max_epochs + 1):
        if DISTILL_ENABLE:
            if ep <= KD_WARMUP_EPOCHS:
                kd_out = 0.0
                kd_feat = 0.0
            elif ep <= KD_WARMUP_EPOCHS + 5:
                kd_out = float(DISTILL_WEIGHT)
                kd_feat = 0.0
            else:
                kd_out = float(DISTILL_WEIGHT)
                kd_feat = float(DISTILL_WEIGHT) * float(KD_FEAT_RATIO)

            tr_loss, _, _, _, _ = _epoch_whole_distill(
                model, ltr, optimizer, device,
                desc=f"{tag} Train ep{ep}",
                studentA=studentA, studentB=studentB, student_head=student_head, student_star=student_star,
                kd_out=kd_out, kd_feat=kd_feat, enable_feat_kd=ENABLE_FEAT_KD
            )
            va_loss, _, _, _, _ = _epoch_whole_distill(
                model, lva, None, device,

                desc=f"{tag}  Val  ep{ep}",
                studentA=studentA, studentB=studentB, student_head=student_head, student_star=student_star,
                kd_out=kd_out, kd_feat=kd_feat, enable_feat_kd=ENABLE_FEAT_KD
            )

        else:
            tr_loss, _, _, _, _ = _epoch_whole(
                model, ltr, optimizer, device,
                desc=f"{tag} Train ep{ep}",
                bound_loss_fn=bound_loss_fn,
                bound_weight=BOUND_WEIGHT
            )
            va_loss, _, _, _, _ = _epoch_whole(model, lva, None, device, desc=f"{tag}  Val  ep{ep}")

        print(f"{tag} ep {ep:02d}: train {tr_loss:.8f} | val {va_loss:.8f} | patience {bad}/{PATIENCE}")
        scheduler.step(va_loss)

        improved = va_loss < best_val - 1e-12
        if improved:
            best_val = va_loss
            bad = 0
            if save_best_path:
                os.makedirs(os.path.dirname(save_best_path), exist_ok=True)
                if DISTILL_ENABLE:
                    best_obj = {
                        "model": model.state_dict(),
                        "studentA": studentA.state_dict(),
                        "studentB": studentB.state_dict(),
                    }
                else:
                    best_obj = model.state_dict()
                torch.save(best_obj, save_best_path)
        else:
            bad += 1

        if ckpt_dir:
            state = {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
                "bad": bad,
                "network": type(model).__name__,
                "distill": bool(DISTILL_ENABLE),
            }
            if DISTILL_ENABLE:
                state["studentA"] = studentA.state_dict()
                state["studentB"] = studentB.state_dict()
            torch.save(state, last_path)
            if keep_every_epoch:
                torch.save(state, os.path.join(ckpt_dir, f"epoch_{ep:03d}.pt"))

        if bad >= PATIENCE:
            print(f"{tag} Early stop.")
            break

    if save_best_path and os.path.isfile(save_best_path):
        ckpt_best = torch.load(save_best_path, map_location=device)
        if DISTILL_ENABLE:
            if isinstance(ckpt_best, dict) and "model" in ckpt_best:
                model.load_state_dict(ckpt_best["model"])
                if "studentA" in ckpt_best and studentA is not None:
                    studentA.load_state_dict(ckpt_best["studentA"])
                if "studentB" in ckpt_best and studentB is not None:
                    studentB.load_state_dict(ckpt_best["studentB"])
            else:
                raise RuntimeError(f"{tag} best.pt has no student weights; please delete old best.pt or retrain.")
        else:
            if isinstance(ckpt_best, dict) and "model" in ckpt_best:
                model.load_state_dict(ckpt_best["model"])
            else:
                model.load_state_dict(ckpt_best)

    if DISTILL_ENABLE:
        _, y_true, y_pred, times, cids = _epoch_whole_distill(
            model, lte, None, device, desc=f"{tag}  Test",
            studentA=studentA, studentB=studentB, student_head=student_head, student_star=student_star,
            kd_out=0.0, kd_feat=0.0, enable_feat_kd=ENABLE_FEAT_KD
        )
    else:
        _, y_true, y_pred, times, cids = _epoch_whole(model, lte, None, device, desc=f"{tag}  Test")

    return y_true, y_pred, times, cids


