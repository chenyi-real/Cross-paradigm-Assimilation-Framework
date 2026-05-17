import os
import sys

# 保证能 import 到项目内模块
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import configs.config_distill_udds as config_distill_udds
from configs.config_distill_udds import (
    DATA_DIR, CSV_FILES, DEVICE,
    WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS
)
from dataloader import load_UDDS_csv_data, build_dataloaders_from_udds_df
from train_distill_udds import _build_teacher_model
from model import MLPStudentEncoder, build_whole_model, SOP_MLPStudentEncoder

from predict_soh_distill_v2 import  predict_soh_for_cid, load_student_model

def soc_results_plot(soc_results, out_dir, fname):
    if soc_results.times is None or soc_results.cids is None:
        raise RuntimeError("times 或 cids 为空，无法按 cid / xlsx 分组，请检查 dataloader。")

    # 整理成 DataFrame
    df = pd.DataFrame({
        "time": soc_results.times,
        "y_true": soc_results.y_true,
        "y_pred": soc_results.y_pred,
        "cid": soc_results.cids
    })

    # 输出根目录： out_dir / fname_D 这种
    base_dir = os.path.join(out_dir, "pred_D_test_SOC", f"fname_{fname}")
    os.makedirs(base_dir, exist_ok=True)

    # 1）保存整体的 raw 结果
    all_csv_path = os.path.join(base_dir, f"{fname}_all_predictions.csv")
    df.to_csv(all_csv_path, index=False)
    print(f"[Save] all predictions csv -> {all_csv_path}")

    # 2）每个 cid 单独保存 csv + png
    uniq_cids = np.unique(df["cid"].values)
    print(f"[Info] total {len(uniq_cids)} cids")

    for cid in uniq_cids:
        sub = df[df["cid"] == cid].sort_values("time")
        cid_int = int(cid)

        cid_csv = os.path.join(base_dir, f"{fname}_cid{cid_int:03d}.csv")
        cid_fig = os.path.join(base_dir, f"{fname}_cid{cid_int:03d}.png")

        sub.to_csv(cid_csv, index=False)
        _save_one_curve_fig(
            sub["time"].values,
            sub["y_true"].values,
            sub["y_pred"].values,
            out_path=cid_fig,
            title=f"{fname} - cid {cid_int}"
        )
        print(f"[Save] cid={cid_int} csv -> {cid_csv}")
        print(f"[Save] cid={cid_int} fig -> {cid_fig}")

    # 3）所有 cid 合在一张图里，使用不同颜色
    all_fig = os.path.join(base_dir, f"{fname}_all_cids.png")
    plt.figure()
    cmap = plt.cm.get_cmap("tab20", len(uniq_cids))

    for idx, cid in enumerate(uniq_cids):
        sub = df[df["cid"] == cid].sort_values("time")
        color = cmap(idx)
        cid_int = int(cid)
        plt.plot(sub["time"].values, sub["y_pred"].values,
                label=f"cid {cid_int}", color=color)

    plt.xlabel("Time")
    plt.ylabel("SOC")
    plt.title(f"{fname} - All cids (Student)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(all_fig, dpi=150)
    plt.close()
    print(f"[Save] merged fig -> {all_fig}")


def sop_results_plot(sop_results, out_dir, fname):
    out_dir = os.path.join(out_dir, "pred_D_test_SOP", f"fname_{fname}")
    os.makedirs(out_dir, exist_ok=True)

    rmse, mae = compute_metrics(sop_results.y_true, sop_results.y_pred)
    print(f"RMSE={rmse:.6f}  MAE={mae:.6f}")

    os.makedirs(out_dir, exist_ok=True)

    if sop_results.cid.size > 0 and sop_results.V.size > 0:
        for c in sorted(np.unique(sop_results.cid)):
            mask = (sop_results.cid == c)
            if not np.any(mask):
                continue

            sop_true = sop_results.y_true[mask] * sop_results.V[mask]
            sop_pred = sop_results.y_pred[mask] * sop_results.V[mask]

            _series_plot(
                times=sop_results.t_end[mask],
                y_true=sop_true,
                y_pred=sop_pred,
                out_png=os.path.join(out_dir, f"cycle_{int(c)}.png"),
                title=f"Cycle {int(c)}"
            )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return rmse, mae

def _series_plot(times, y_true, y_pred, out_png, title):
    order = np.argsort(times)
    t = np.asarray(times)[order]
    yt = np.asarray(y_true)[order]
    yp = np.asarray(y_pred)[order]

    plt.figure()
    plt.plot(t, yt, label="y_true")
    plt.plot(t, yp, label="y_pred")
    plt.xlabel("t_end (s)")
    plt.ylabel("SOP")
    plt.title(title)
    plt.legend()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[Saved] {out_png}")


class SOCResults():
    def __init__(self) -> None:
        self.ys, self.yps, self.times_all, self.cids_all = [], [], [], []
        self.y_true = None
        self.y_pred = None
        self.times = None
        self.cids = None

class SOPResults():
    def __init__(self) -> None:
        self.Ys, self.YPs, self.Ts, self.CIDs, self.Vs = [], [], [], [], []
        self.y_true = None
        self.y_pred = None
        self.t_end = None
        self.cid = None
        self.V = None

def _build_student_only(device=DEVICE):
    """
    只构建 student 模型，但需要 teacher 的 encoder 维度来确定 hidden_size。
    """
    teacher = _build_teacher_model(device)
    dim_teacher = teacher.return_encoder_dim()
    
    del teacher
    student = MLPStudentEncoder(in_dim=3, hidden_size=dim_teacher).to(device)
    return student


def _load_test_df_for_fname(fname: str):
    """
    复用 main_distill_udds.py 里的划分逻辑，只返回 test_df
    """
    if fname == "A":
        val_csv = ["Cycling_1", "Cycling_2"]
        test_csv = ["Cycling_3"]
        train_csv = [x for x in CSV_FILES if x not in val_csv and x not in test_csv]
    elif fname == "B":
        val_csv = ["Cycling_1", "Cycling_2"]
        test_csv = ["Cycling_4"]
        train_csv = [x for x in CSV_FILES if x not in val_csv and x not in test_csv]
    elif fname == "C":
        val_csv = ["Cycling_1", "Cycling_2"]
        test_csv = [f"Cycling_{i}" for i in range(8, 12)]
        train_csv = [x for x in CSV_FILES if x not in val_csv and x not in test_csv]
    else:  # fname == 'D' 或 其他默认走 D 的设置
        val_csv = ["Cycling_1", "Cycling_2"]
        test_csv = [f"Cycling_{i}" for i in range(12, 15)]
        train_csv = [x for x in CSV_FILES if x not in val_csv and x not in test_csv]

    print(f"[Info] fname={fname}, test_csv={test_csv}")
    test_df = load_UDDS_csv_data(DATA_DIR, test_csv, fname, split="test")
    return test_df


def _predict_student_on_dataloader(args, soh_model, student_model, sop_student_model_A, sop_student_model_B, SOP_teacher_model, 
                                   loader, device=DEVICE):
    """
    只用 student 模型前向，收集 y_true / y_pred / time / cid
    """
    # 模型加载
    # 这里还需要加载SOH模型
    # SOH模型也需要加载数据，那么SOC那边专属的
    soh_model.eval()
    student_model.eval()
    sop_student_model_A.eval()
    sop_student_model_B.eval()
    SOP_teacher_model.eval()
    # SOC 结果
    soc_results = SOCResults()
    sop_results = SOPResults()
    # ys, yps, times_all, cids_all = [], [], [], []


    pbar = tqdm(loader, desc="[Predict] student", leave=False)
    with torch.no_grad():
        history_cid = None
        for batch in pbar:
            tb = None
            cids = None

            # 兼容 dataloader 的多种返回格式
            if isinstance(batch, (list, tuple)):
                if len(batch) == 4:
                    xb, yb, tb, cids = batch
                elif len(batch) == 3:
                    # xb, yb, third = batch
                    soc_out, sop_out, third = batch
                    xb, yb, tb = soc_out
                    xA, xB, y, t_end = sop_out
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

            # SOH推理环节
            # 最开始需要先推理，得到cids==12的 SOH
            # 接下来需要做判断 当这个时候的cids出现13的时候，
            if history_cid is None:
                # 运行SOH推理   
                history_cid = cids[-1]
                soh = predict_soh_for_cid(args,
                              model=soh_model,
                              cid=history_cid,
                              dataset_root='./data/UDDS_HIS',
                              normalization=True)

                print(f"CID: {history_cid}, SOH: {soh}")
                
            if cids[-1] != history_cid:
                # 运行 SOH模型
                history_cid = cids[-1]
                soh = predict_soh_for_cid(args,
                              model=soh_model,
                              cid=history_cid,
                              dataset_root='./data/UDDS_HIS',
                              normalization=True)

                print(f"CID: {history_cid},  SOH: {soh}")

            # SOC推理环节
            xb, yb = xb.to(device), yb.to(device) # soc data

            # soc student 输出 (pred, feat)
            stu_y, _ = student_model(xb)
            stu_y = torch.sigmoid(stu_y)

            soc_results.ys.append(yb.detach().cpu().numpy().reshape(-1))
            soc_results.yps.append(stu_y.detach().cpu().numpy().reshape(-1))

            if tb is not None:
                try: 
                    soc_results.times_all.append(tb.detach().cpu().numpy().reshape(-1))
                except Exception:
                    soc_results.times_all.append(np.asarray(tb).reshape(-1))
            if cids is not None:
                try:
                    soc_results.cids_all.append(cids.detach().cpu().numpy().reshape(-1))
                except Exception:
                    soc_results.cids_all.append(np.asarray(cids).reshape(-1))

            # SOP 推理环节
            xA, xB, y = xA.to(device), xB.to(device), y.to(device) # sop data
            xA[:,-1,1] = stu_y
            v_last = xA[..., 0][:, -1]          # [B]
            sop_results.Vs.append(v_last.cpu().numpy().reshape(-1))

            sA = sop_student_model_A(xA)
            sB = sop_student_model_B(xB)
            _, _, sA2, sB2 = SOP_teacher_model.star(sA, sB)
            y_pred = SOP_teacher_model.head(torch.cat([sA2, sB2], dim=-1)).squeeze(-1)

            sop_results.Ys.append(y.cpu().numpy().reshape(-1))
            sop_results.YPs.append(y_pred.cpu().numpy().reshape(-1))
            try:
                sop_results.Ts.append(t_end.cpu().numpy().reshape(-1))
            except Exception:
                sop_results.Ts.append(np.asarray(t_end).reshape(-1))
            

    # soc results
    soc_results.y_true = np.concatenate(soc_results.ys) if soc_results.ys else np.array([])
    soc_results.y_pred = np.concatenate(soc_results.yps) if soc_results.yps else np.array([])
    soc_results.times = np.concatenate(soc_results.times_all) if soc_results.times_all else None
    soc_results.cids = np.concatenate(soc_results.cids_all) if soc_results.cids_all else None

    # sop reuslts
    sop_results.y_true = np.concatenate(sop_results.Ys) if sop_results.Ys else np.array([])
    sop_results.y_pred = np.concatenate(sop_results.YPs) if sop_results.YPs else np.array([])
    sop_results.t_end = np.concatenate(sop_results.Ts) if sop_results.Ts else np.array([])
    sop_results.cid = np.concatenate(soc_results.cids_all) if soc_results.cids_all else np.array([])
    sop_results.V = np.concatenate(sop_results.Vs) if sop_results.Vs else np.array([])
    return soc_results, sop_results
    # return y_true, y_pred, times, cids


def _save_one_curve_fig(times, y_true, y_pred, out_path, title=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(times, y_true, label="True")
    plt.plot(times, y_pred, label="Pred")
    if title:
        plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("SOC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def predict_and_export(args, fname: str, soh_ckpt:str, student_ckpt: str, sop_ckpt: str, out_dir: str,):
    """
    主函数：
    - 根据 fname 读取 test_df
    - 构建 dataloader
    - 加载学生模型做预测
    - 每个 cid 单独保存 csv + png
    - 再做一张所有 cid 的大图（不同颜色）
    - 所有结果都保存在 out_dir/fname_xxx 下
    """
    import torch  # 放在函数里，避免脚本 import 顺序问题

    # ====== 数据 ======
    test_df = _load_test_df_for_fname(fname)
        # 到这一步，所有的数据已经读取好了，我需要重新写
    _, _, lte = build_dataloaders_from_udds_df(
        test_df, test_df, test_df,
        WINDOW, STRIDE, BATCH_SIZE, NUM_WORKERS, if_demo=True
    )
    
    # ====== 学生模型 ======
    # 加载 SOH student model
    soh_model = load_student_model(soh_ckpt)
    # 加载 SOC student model
    student = _build_student_only(device=DEVICE)
    print(f"[Load] student checkpoint: {student_ckpt}")
    ckpt = torch.load(student_ckpt, map_location=DEVICE)
    student.load_state_dict(ckpt, strict=False)

    # 加载 SOP student model
    model = build_whole_model().to(DEVICE).eval()
    dimA = int(model.encoder_a.out_dim)
    dimB = int(model.encoder_b.out_dim)
    studentA = SOP_MLPStudentEncoder(in_dim=3, hidden_size=dimA).to(DEVICE).eval()
    studentB = SOP_MLPStudentEncoder(in_dim=3, hidden_size=dimB).to(DEVICE).eval()
    
    obj = torch.load(sop_ckpt, map_location=DEVICE)
    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"], strict=False)
        if "studentA" in obj:
            studentA.load_state_dict(obj["studentA"], strict=False)
        if "studentB" in obj:
            studentB.load_state_dict(obj["studentB"], strict=False)
    else:
        model.load_state_dict(obj, strict=False)
        print("[Warn] checkpoint missing students; distilled inference may be invalid.")
    
    
    # ====== 前向预测 ======
    # y_true, y_pred, times, cids = _predict_student_on_dataloader(student, studentA, studentB, model, lte, device=DEVICE)
    
    soc_results, sop_results = _predict_student_on_dataloader(args, soh_model, student, studentA, studentB, model, lte, device=DEVICE)


    soc_results_plot(soc_results=soc_results, out_dir=out_dir, fname=fname)
    sop_results_plot(sop_results, out_dir, fname=fname)
    # sop results save


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, default='D',
                        help="数据划分标识，如 A / B / C / D")
    parser.add_argument("--soh_student_ckpt", type=str, default='./distill_student_weights/soh_best_student_distill.pth',
                        help="best_student_distill.pth 的路径")
    parser.add_argument("--soc_student_ckpt", type=str, default='./distill_student_weights/soc_best_student_distill.pth',
                        help="best_student_distill.pth 的路径")
    parser.add_argument("--sop_student_ckpt", type=str, default='./distill_student_weights/sop_best_student_distill.pt',
                        help="best_student_distill.pth 的路径")
    parser.add_argument("--out_dir", type=str, default='./runs_predict/',
                        help="所有 csv 和 png 的输出根目录")
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='assemble_test', help='save folder')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    args = parser.parse_args()

    predict_and_export(
        args, 
        fname=args.fname,
        soh_ckpt=args.soh_student_ckpt,
        student_ckpt=args.soc_student_ckpt,
        sop_ckpt=args.sop_student_ckpt,
        out_dir=args.out_dir
    )

