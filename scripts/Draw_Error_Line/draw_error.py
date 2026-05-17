
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot prediction scatter (true vs pred) with error colormap, auto-adaptive axes.

Big tasks: baseline / transfer / distill
Subtasks: SOH / SOC / SOP

Each item root requires:
  - pred_label.npy
  - true_label.npy

New:
  - Fine-grained labels: dataset (NASA/CALCE/MIT) + battery group (optional)
"""

import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt


from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, ScalarFormatter

# Optional style
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science"])
    plt.rcParams['font.family'] = ["Times New Roman"]

except Exception:
    pass

# 只对 baseline + SOP + MIT 过滤（示例：过滤掉最极端 1% 的误差点）
FILTER_PERCENTILE = {
    ("baseline", "SOP", "MIT"): 99.8,   # 你可以改成 98/99.5 等
}


# ---- SOH level mapping for transfer / distill ----
SOH_LEVEL_MAP = {
    "A_results": r"98\%-97\%",
    "B_results": r"97\%-96\%",
    "C_results": r"92\%-91\%",
    "D_results": r"91\%-90\%",
}



# ===== Global font size control =====
FONT_SIZE = 16          # 你想要的“基准字号”，比如 10 / 11 / 12

plt.rcParams.update({
    "font.size": FONT_SIZE,          # 默认文本
    "axes.titlesize": FONT_SIZE + 1, # 子图标题
    "axes.labelsize": FONT_SIZE,     # x/y label
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.titlesize": FONT_SIZE + 2,
})
# ---- Axes (box) linewidth: 控制四个边框的粗细 ----
plt.rcParams["axes.linewidth"] = 1.2 # 1.2




SOH_MAX = 0.9884384870529175
SOH_MIN = 0.9072180986404419


# -------------------------
# 1) Select quantities
# -------------------------
SELECT_QUANTITIES = {"SOH","SOC","SOP"}  # {"SOH","SOC","SOP"} / {"SOC"} / {"SOP"}

# -------------------------
# 2) Items
# -------------------------
ITEMS = [
    # ===== SOH =====
    # ===== NASA （分电池） =====
    dict(
        big_task="baseline",
        quantity="SOH",
        dataset="NASA",
        battery="B0005",
        root="./exp_20250903_table4/results of PINNdebug/NASA results-B0005/Experiment8",
    ),
    dict(
        big_task="baseline",
        quantity="SOH",
        dataset="NASA",
        battery="B0006",
        root="./exp_20250903_table4/results of PINNdebug/NASA results-B0006/Experiment8",
    ),
    dict(
        big_task="baseline",
        quantity="SOH",
        dataset="NASA",
        battery="B0007",
        root="./exp_20250903_table4/results of PINNdebug/NASA results-B0007/Experiment4",
    ),
    dict(
        big_task="baseline",
        quantity="SOH",
        dataset="NASA",
        battery="B0018",
        root="./exp_20250903_table4/results of PINNdebug/NASA results-B0018/Experiment6",
    ),
    # ===== CALCE （分电池）=====
    dict(
        big_task="baseline",
        quantity="SOH",
        dataset="CALCE",
        battery="CS2_35",
        root="./exp_20250903_table4/results of PINNdebug/CS2 results-CS2_35/Experiment4",
    ),
    dict(
        big_task="baseline",
        quantity="SOH",
        dataset="CALCE",
        battery="CS2_36",
        root="./exp_20250903_table4/results of PINNdebug/CS2 results-CS2_36/Experiment3",
    ),
    dict(
        big_task="baseline",
        quantity="SOH",
        dataset="CALCE",
        battery="CS2_37",
        root="./exp_20250903_table4/results of PINNdebug/CS2 results-CS2_37/Experiment6",
    ),
    dict(
        big_task="baseline",
        quantity="SOH",
        dataset="CALCE",
        battery="CS2_38",
        root="./exp_20250903_table4/results of PINNdebug/CS2 results-CS2_38/Experiment9",
    ),
     # ===== MIT =====
    dict(
        big_task="baseline",
        quantity="SOH",
        dataset="MIT",
        battery=None,  # MIT 不分组 -> None
        root="./exp_20250903_table4/results of PINNv2/MIT results/Experiment6",
        # title 可不写，自动生成更规范；如果你想强行指定，也可以写 title="..."
    ),
    # ===== transfer 示例 =====
    dict(
        big_task="transfer",
        quantity="SOH",
        dataset="UDDS",
        battery=None,  # MIT 不分组 -> None
        root="./exp_20260101_Transfer_MIT_UDDS_2/MIT-UDDS-Ours/Experiment8",
    ),

    # ===== distill 示例 =====
    dict(
        big_task="distill",
        quantity="SOH",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery=None,
        root="./exp_20260101_distill_v3/UDDS-Ours/Experiment4",
    ),

    # ===== SOC =====
    # ===== baseline =====
    dict(
        big_task="baseline",
        quantity="SOC",
        dataset="NASA",
        battery="B0005",
        root="./test_results/baseline/NASA/B0005",
    ),
    dict(
        big_task="baseline",
        quantity="SOC",
        dataset="NASA",
        battery="B0006",
        root="./test_results/baseline/NASA/B0006",
    ),
    dict(
        big_task="baseline",
        quantity="SOC",
        dataset="NASA",
        battery="B0007",
        root="./test_results/baseline/NASA/B0007",
    ),
    dict(
        big_task="baseline",
        quantity="SOC",
        dataset="NASA",
        battery="B0018",
        root="./test_results/baseline/NASA/B0018",
    ),
    # ===== CALCE （分电池）=====
    dict(
        big_task="baseline",
        quantity="SOC",
        dataset="CALCE",
        battery="CS2_35",
        root="./test_results/baseline/CS2/CS2_35",
    ),
    dict(
        big_task="baseline",
        quantity="SOC",
        dataset="CALCE",
        battery="CS2_36",
        root="./test_results/baseline/CS2/CS2_36",
    ),
    dict(
        big_task="baseline",
        quantity="SOC",
        dataset="CALCE",
        battery="CS2_37",
        root="./test_results/baseline/CS2/CS2_37",
    ),
    dict(
        big_task="baseline",
        quantity="SOC",
        dataset="CALCE",
        battery="CS2_38",
        root="./test_results/baseline/CS2/CS2_38",
    ),
    # ===== MIT =====
    dict(
        big_task="baseline",
        quantity="SOC",
        dataset="MIT",
        battery=None,  # MIT 不分组 -> None
        root="./test_results/baseline/MIT/2017-05-12",
        # title 可不写，自动生成更规范；如果你想强行指定，也可以写 title="..."
    ),


    # ===== transfer =====
    dict(
        big_task="transfer",
        quantity="SOC",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="A_results",
        root="./test_results/transfer/UDDS/run_1/A",
    ),
    dict(
        big_task="transfer",
        quantity="SOC",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="B_results",
        root="./test_results/transfer/UDDS/run_1/B",
    ),
    dict(
        big_task="transfer",
        quantity="SOC",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="C_results",
        root="./test_results/transfer/UDDS/run_1/C",
    ),
    dict(
        big_task="transfer",
        quantity="SOC",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="D_results",
        root="./test_results/transfer/UDDS/run_1/D",
    ),

    # ===== distill =====
    dict(
        big_task="distill",
        quantity="SOC",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="A_results",
        root="./test_results/distill/UDDS/run_1/A",
    ),
    dict(
        big_task="distill",
        quantity="SOC",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="B_results",
        root="./test_results/distill/UDDS/run_1/B",
    ),
    dict(
        big_task="distill",
        quantity="SOC",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="C_results",
        root="./test_results/distill/UDDS/run_1/C",
    ),
    dict(
        big_task="distill",
        quantity="SOC",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="D_results",
        root="./test_results/distill/UDDS/run_1/D",
    ),


    # ===== SOP =====
    # ===== baseline =====
    dict(
        big_task="baseline",
        quantity="SOP",
        dataset="NASA",
        battery="B0005",
        root="./sop_data/baseline/NASA/B0005",
    ),
    dict(
        big_task="baseline",
        quantity="SOP",
        dataset="NASA",
        battery="B0006",
        root="./sop_data/baseline/NASA/B0006",
    ),
    dict(
        big_task="baseline",
        quantity="SOP",
        dataset="NASA",
        battery="B0007",
        root="./sop_data/baseline/NASA/B0007",
    ),
    dict(
        big_task="baseline",
        quantity="SOP",
        dataset="NASA",
        battery="B0018",
        root="./sop_data/baseline/NASA/B0018",
    ),
    # ===== CALCE （分电池）=====
    dict(
        big_task="baseline",
        quantity="SOP",
        dataset="CALCE",
        battery="CS2_35",
        root="./sop_data/baseline/CALCE/CS2_35",
    ),
    dict(
        big_task="baseline",
        quantity="SOP",
        dataset="CALCE",
        battery="CS2_36",
        root="./sop_data/baseline/CALCE/CS2_36",
    ),
    dict(
        big_task="baseline",
        quantity="SOP",
        dataset="CALCE",
        battery="CS2_37",
        root="./sop_data/baseline/CALCE/CS2_37",
    ),
    dict(
        big_task="baseline",
        quantity="SOP",
        dataset="CALCE",
        battery="CS2_38",
        root="./sop_data/baseline/CALCE/CS2_38",
    ),
     # ===== MIT =====
    dict(
        big_task="baseline",
        quantity="SOP",
        dataset="MIT",
        battery=None,  # MIT 不分组 -> None
        root="./sop_data/baseline/MIT",
        # title 可不写，自动生成更规范；如果你想强行指定，也可以写 title="..."
    ),
    # ===== transfer =====
    dict(
        big_task="transfer",
        quantity="SOP",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="A_results",
        root=".sop_data/transfer/A/run_1",
    ),
    dict(
        big_task="transfer",
        quantity="SOP",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="B_results",
        root="./sop_data/transfer/B/run_1",
    ),
    dict(
        big_task="transfer",
        quantity="SOP",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="C_results",
        root="./sop_data/transfer/C/run_1",
    ),
    dict(
        big_task="transfer",
        quantity="SOP",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="D_results",
        root="./sop_data/transfer/D/run_1",
    ),

    # ===== distill =====
    dict(
        big_task="distill",
        quantity="SOP",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="A_results",
        root="./sop_data/distill/A",
    ),
    dict(
        big_task="distill",
        quantity="SOP",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="B_results",
        root="./sop_data/distill/B",
    ),
    dict(
        big_task="distill",
        quantity="SOP",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="C_results",
        root="./sop_data/distill/C",
    ),
    dict(
        big_task="distill",
        quantity="SOP",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="D_results",
        root="./sop_data/distill/D",
    ),
    


    # 继续加：
    # dict(big_task="transfer", quantity="SOC", dataset="NASA", battery="B0007", root="..."),
    # dict(big_task="baseline", quantity="SOP", dataset="CALCE", battery="CS2_38", root="..."),
]



_ITEMS = [
    dict(
        big_task="baseline",
        quantity="SOP",
        dataset="MIT",
        battery=None,  # MIT 不分组 -> None
        root="./sop_data/baseline/MIT",
        # title 可不写，自动生成更规范；如果你想强行指定，也可以写 title="..."
    ),
    dict(
        big_task="distill",
        quantity="SOP",
        dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
        battery="D_results",
        root="./sop_data/distill/D",
    ),
    # dict(
    #     big_task="distill",
    #     quantity="SOH",
    #     dataset="UDDS",     # 如果你这里不是 NASA/CALCE/MIT 也没关系
    #     battery=None,
    #     root=f"./exp_20260101_distill_v3/UDDS-Ours/Experiment{i}",
    # )
    # for i in range(1,11) 
]


# -------------------------
# Plot settings
# -------------------------
PRED_NAME = "pred_label.npy"
TRUE_NAME = "true_label.npy"

# Error colorbar range:
# - None: auto by global max error
# - (0, 0.1): fixed range
ERROR_RANGE = None

FORCE_EQUAL_ASPECT = True
SCATTER_SIZE = 12 # 10
SCATTER_ALPHA = 0.75 # 0.75

DIAG_STYLE = dict(linestyle="--", linewidth=1.0, alpha=1.0)
DIAG_COLOR = "#ff4d4e"

# Save settings
SAVE_DIR = "./HD_scatter_out"          # 每个小图输出目录
SAVE_EACH = True                    # 单独保存每个 item
SAVE_COMBINED = True                # 也保存合并总图（可关掉）
OUT_SVG = "HD_estimation_results.svg"
OUT_PNG = "HD_estimation_results.png"
DPI = 600 # 600

# Ticks tuning
N_TICKS = 5                         # 默认目标刻度数
NARROW_RANGE_THRESHOLD = 0.02       # span 小于这个值时，按“窄范围”策略（更细刻度/更多小数）


def _load_pair(root: str):
    pred_path = os.path.join(root, PRED_NAME)
    true_path = os.path.join(root, TRUE_NAME)
    if not (os.path.exists(pred_path) and os.path.exists(true_path)):
        return None, None

    
    pred = np.load(pred_path).astype(np.float32).reshape(-1)
    true = np.load(true_path).astype(np.float32).reshape(-1)

    m = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[m], true[m]
    if pred.size == 0:
        return None, None
    return pred, true




def _format_title(it: dict) -> str:
    # 手动 title 优先
    if "title" in it and it["title"]:
        return it["title"]

    big_task = str(it.get("big_task", "")).strip().lower()
    quantity = str(it.get("quantity", "")).strip()
    dataset = str(it.get("dataset", "")).strip()
    battery = it.get("battery", None)
    battery = None if battery is None else str(battery).strip()

    # =========================
    # 1) baseline
    # =========================
    if big_task == "baseline":
        if dataset and quantity:
            if battery:
                return f"{dataset}: {quantity} ({battery})"
            else:
                return f"{dataset}: {quantity}"
        return dataset or quantity or ""

    # =========================
    # 2) transfer
    # =========================
    if big_task == "transfer":
        head = f"Transfer learning: {quantity}"

        # SOH：单行
        if quantity == "SOH":
            return head

        # SOC / SOP：两行
        if battery in SOH_LEVEL_MAP:
            level = SOH_LEVEL_MAP[battery]
            return f"{head}\n(SOH level: {level})"
        else:
            return head

    # =========================
    # 3) distill
    # =========================
    if big_task == "distill":
        head = f"Knowledge distillation: {quantity}"

        # SOH：单行
        if quantity == "SOH":
            return head

        # SOC / SOP：两行
        if battery in SOH_LEVEL_MAP:
            level = SOH_LEVEL_MAP[battery]
            return f"{head}\n(SOH level: {level})"
        else:
            return head

    # =========================
    # fallback（理论上不会走到）
    # =========================
    return quantity or ""





def _safe_stem(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^0-9A-Za-z_\-\.]+", "", s)
    return s.strip("_") or "item"


def _item_filename(it: dict) -> str:
    big = str(it.get("big_task", "task")).strip()
    qty = str(it.get("quantity", "Q")).strip()
    ds = str(it.get("dataset", "DS")).strip()
    bat = it.get("battery", None)
    bat = "" if bat is None else str(bat).strip()
    parts = [big, qty, ds]
    if bat:
        parts.append(bat)
    return _safe_stem("__".join(parts))


def _auto_limits(x, y, pad_ratio=0.06, tiny_span_eps=1e-12):
    """
    真正按数据范围自适应：
      - 覆盖 x & y 的 min/max
      - 加一点 margin
      - 只有在“几乎常数”的极端情况下才给一个非常小的 span，避免 lim 相等报错
    """
    vmin = float(min(x.min(), y.min()))
    vmax = float(max(x.max(), y.max()))
    span = vmax - vmin

    if span < tiny_span_eps:
        # 极端：几乎常数，给一个非常小的范围
        center = 0.5 * (vmin + vmax)
        span = max(1e-6, abs(center) * 1e-6)
        vmin = center - 0.5 * span
        vmax = center + 0.5 * span

    pad = span * float(pad_ratio)
    return (vmin - pad, vmax + pad)


def _choose_decimals(span: float) -> int:
    """
    span 越小，小数位越多；保证像 0.905~0.91 这种能显示 3~4 位小数。
    """
    if span <= 0:
        return 4
    # 让“刻度间隔”大约是 span/(N_TICKS-1)，然后决定显示多少位
    step = span / max(1, (N_TICKS - 1))
    if step <= 0:
        return 4
    # 小数位 = ceil(-log10(step)) + 1，限制在 [2, 6]
    dec = int(math.ceil(-math.log10(step))) + 1
    return int(np.clip(dec, 2, 6))


def _apply_adaptive_ticks(ax, lim):
    """
    解决你说的“刻度太大/不细”的问题：
      - 禁用 offset/scientific
      - 窄范围时用更多小数
      - 控制主刻度数量
    """
    lo, hi = lim
    span = float(hi - lo)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=N_TICKS))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=N_TICKS))

    # 禁用科学计数法 & offset
    sf = ScalarFormatter(useOffset=False)
    sf.set_scientific(False)
    ax.xaxis.set_major_formatter(sf)
    ax.yaxis.set_major_formatter(sf)

    if span < NARROW_RANGE_THRESHOLD:
        dec = _choose_decimals(span)
        fmt = FormatStrFormatter(f"%.{dec}f")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)


def _get_cmap():
    # 你原来的自定义 cmap
    color_list = ['#74AED4', '#7BDFF2', '#FBDD85', '#F46F43', '#CF3D3E']
    return plt.cm.colors.LinearSegmentedColormap.from_list("custom_cmap", color_list, N=256) # type: ignore


def plot_items(items, select_quantities):
    items = [it for it in items if it.get("quantity") in select_quantities]
    if len(items) == 0:
        raise RuntimeError(f"No items to plot. SELECT_QUANTITIES={select_quantities}, ITEMS={len(ITEMS)}")

    # ---- progress helper ----
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(items), desc="Plotting", dynamic_ncols=True)
        use_tqdm = True
    except Exception:
        pbar = None
        use_tqdm = False
    loaded = []
    all_err = []

    # 先加载，顺便给进度（加载阶段）
    for i, it in enumerate(items, 1):
        pred, true = _load_pair(it["root"])
        if pred is None:
            msg = f"[skip] missing/empty: {it['root']}"
            print(msg)
            if use_tqdm:
                pbar.update(1)
            continue
        if (it['big_task'] == 'transfer' or it['big_task'] == 'distill') and it['quantity'] == 'SOH':
            pred = (pred + 1) / 2 * (SOH_MAX - SOH_MIN) + SOH_MIN
            true = (true + 1) / 2 * (SOH_MAX - SOH_MIN) + SOH_MIN


        # err = np.abs(pred - true)
        # loaded.append((it, pred, true, err))
        # all_err.append(err)


        err_raw = np.abs(pred - true)

        # 1) colorbar 仍用“原始全量误差”
        all_err.append(err_raw)

        # 2) 按条件决定是否过滤（只影响画点，不影响 colorbar）
        key = (it.get("big_task"), it.get("quantity"), it.get("dataset"))
        p = FILTER_PERCENTILE.get(key, None)

        if p is not None:
            thr = np.percentile(err_raw, p)
            keep = err_raw <= thr
            filtered_ratio = 1.0 - (keep.sum() / max(1, keep.size))
            print(f"[filter] {_item_filename(it)} | p{p} thr={thr:.6g} | removed={filtered_ratio*100:.2f}%")

            pred_plot = pred[keep]
            true_plot = true[keep]
            err_plot  = err_raw[keep]
        else:
            pred_plot = pred
            true_plot = true
            err_plot  = err_raw

        loaded.append((it, pred_plot, true_plot, err_plot))




        if use_tqdm:
            pbar.update(1)
        else:
            print(f"[load] {i}/{len(items)}: {_item_filename(it)}")

    if use_tqdm:
        pbar.close()

    if len(loaded) == 0:
        raise RuntimeError("All items are skipped (missing npy or empty after filtering).")

    # ---- error normalization (global) ----
    all_err = np.concatenate(all_err, axis=0)
    if ERROR_RANGE is None:
        e_min = float(np.min(all_err))
        e_max = float(np.max(all_err))
        if math.isclose(e_min, e_max):
            e_max = e_min + 1e-6
        err_norm = Normalize(vmin=e_min, vmax=e_max)
    else:
        err_norm = Normalize(vmin=float(ERROR_RANGE[0]), vmax=float(ERROR_RANGE[1]))

    cmap = _get_cmap()

    # -----------------------------
    # A) Save each item separately
    # -----------------------------
    if SAVE_EACH:
        os.makedirs(SAVE_DIR, exist_ok=True)

        try:
            from tqdm import tqdm
            pbar2 = tqdm(total=len(loaded), desc="Saving each plot", dynamic_ncols=True)
            use_tqdm2 = True
        except Exception:
            pbar2 = None
            use_tqdm2 = False

        for idx, (it, pred, true, err) in enumerate(loaded, 1):
            fig = plt.figure(figsize=(4.2, 3.6), dpi=DPI)
            ax = fig.add_subplot(111)

            ax.scatter(true, pred, c=err, cmap=cmap, norm=err_norm,
                       s=SCATTER_SIZE, alpha=SCATTER_ALPHA, edgecolors="none")

            lim = _auto_limits(true, pred, pad_ratio=0.06)
            ax.plot([lim[0], lim[1]], [lim[0], lim[1]], color=DIAG_COLOR, **DIAG_STYLE)

            ax.set_xlim(lim)
            ax.set_ylim(lim)
            if FORCE_EQUAL_ASPECT:
                ax.set_aspect("equal", adjustable="box")

            _apply_adaptive_ticks(ax, lim)

            qty = it.get("quantity", "")
            ax.set_xlabel(f"True {qty}")
            ax.set_ylabel("Prediction")
            ax.set_title(_format_title(it))
            ax.grid(True, linewidth=0.3, alpha=0.35)

            # colorbar for each plot (可选：你想更干净也可以关掉)
            sm = ScalarMappable(cmap=cmap, norm=err_norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.04)
            cb.set_label("Absolute error")

            stem = _item_filename(it)
            out_svg = os.path.join(SAVE_DIR, f"{stem}.svg")
            out_png = os.path.join(SAVE_DIR, f"{stem}.png")
            fig.tight_layout()
            # fig.savefig(out_svg, format="svg")
            fig.savefig(out_png, format="png")
            plt.close(fig)

            if use_tqdm2:
                pbar2.update(1)
            else:
                print(f"[save] {idx}/{len(loaded)} -> {out_png}")

        if use_tqdm2:
            pbar2.close()

    # -----------------------------
    # B) Save combined figure
    # -----------------------------
    if SAVE_COMBINED:
        # +1 cell for colorbar
        n = len(loaded)
        n_total = n  # n + 1

        ncols = 5 if n_total >= 20 else (4 if n_total >= 8 else (3 if n_total >= 5 else 2))      
        nrows = int(math.ceil(n_total / ncols))

        fig, axs = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.8 * nrows), dpi=DPI)
        axs = np.array(axs).reshape(-1)

        for idx, (it, pred, true, err) in enumerate(loaded):
            ax = axs[idx]
            ax.scatter(true, pred, c=err, cmap=cmap, norm=err_norm,
                       s=SCATTER_SIZE, alpha=SCATTER_ALPHA, edgecolors="none")

            lim = _auto_limits(true, pred, pad_ratio=0.06)
            ax.plot([lim[0], lim[1]], [lim[0], lim[1]], color=DIAG_COLOR, **DIAG_STYLE)

            ax.set_xlim(lim)
            ax.set_ylim(lim)
            if FORCE_EQUAL_ASPECT:
                ax.set_aspect("equal", adjustable="box")

            _apply_adaptive_ticks(ax, lim)

            qty = it.get("quantity", "")
            ax.set_xlabel(f"True {qty}")
            ax.set_ylabel("Prediction")
            ax.set_title(_format_title(it))
            ax.grid(True, linewidth=0.3, alpha=0.35)

        # # colorbar axis
        # cax = axs[n]
        # sm = ScalarMappable(cmap=cmap, norm=err_norm)
        # sm.set_array([])
        # cb = fig.colorbar(sm, cax=cax)
        # cb.set_label("Absolute error")

        # for j in range(n + 1, len(axs)):
        #     axs[j].axis("off")

        fig.tight_layout()
        # fig.savefig(OUT_SVG, format="svg")
        fig.savefig(OUT_PNG, format="png")
        plt.show()


if __name__ == "__main__":
    plot_items(ITEMS, SELECT_QUANTITIES)

