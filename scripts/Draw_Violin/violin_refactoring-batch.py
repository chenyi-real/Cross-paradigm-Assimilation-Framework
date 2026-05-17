'''
当前v5版本是分batch画出来的, 每个batch分开的图更能展示不同电池/Cell之间的分布差异, 但缺点是每个图的数据量更少了, 可能会有点稀疏. 可以根据实际情况选择用哪个版本的代码来画图.
'''
import os
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])
plt.rcParams['font.family'] = ["Times New Roman"]

import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FormatStrFormatter

# ==============================
# 全局常量配置
# ==============================

# ==============================
# 任务类型：'baseline' 或 'transfer' 或 'distill' 
# ==============================
TASK = 'transfer'  # 改成 'transfer' 即可运行 transfer 任务


FONT_SIZE = 18 # 14

DATA_TABLE = {
    'NASA': 'NASA',
    'CS2': 'CALCE',
    'MIT': 'MIT',
    'MIT-UDDS': 'MIT Transfer to UDDS',
}

METHOD_TABLE = {
    'method1': "W/O ANN branch",
    'method2': "W/O SNN branch",
    'method3': "W/O HCO",
    'method4': "W/O PINN branch",
}

# 使用到的模型名称
BASELINE_METHODS = ['Ours', 'MLP', 'CNN', 'LSTM', 'TCN', 'PINN']
DISTILL_METHODS = ['Distilled student', 'Teacher', 'Non-distilled student']

# 默认先用 baseline 的方法（baseline / transfer 共用）

if TASK =='distill':
    METHODS = DISTILL_METHODS
else:
    METHODS = BASELINE_METHODS

# 数据集对应的列名
DATASETS = ['NASA', 'CS2', 'MIT']
DATA_TEST_IDS = {
    'NASA': ['B0005', 'B0006', 'B0007', 'B0018'],
    'CS2': ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38'],
    # MIT 这里是指标类（同时有 MAE / RMSE）
    'MIT': ['MAE', 'RMSE'],
}

# 颜色
# COLORS = ["#F07474", "#45A8C0", "#ADCFEB", "#1BB89C", "#B7A6D2"][:len(METHODS)]
# 新颜色
COLORS = ["#F77676", "#45A8C0", "#ADCFEB", "#1BB89C", "#B7A6D2", "#FFEBB9"][:len(METHODS)]



ROOT = '20260313_exp_data_sum'  # 结果根目录，根据实际情况修改
# ==============================
# baseline 三个 quantity 的路径配置
# ==============================

BASELINE_CONFIG = {
    # ====== 已经有的 SOH 配置（沿用你原来的路径） ======
    'SOH': {
        'rmse_root': f"./{ROOT}/SOH-base-RMSE",        # SOH-RMSE 结果目录
        'mae_root': f"./{ROOT}/SOH-base-MAE",     # SOH-MAE 结果目录
    },

    # ====== 你填上 SOC 的结果路径 ======
    'SOC': {
        'rmse_root': f"./{ROOT}/SOC-base-RMSE",                # TODO: SOC-RMSE 结果目录
        'mae_root': f"./{ROOT}/SOC-base-MAE",                  # TODO: SOC-MAE 结果目录
    },

    # ====== 你填上 SOP 的结果路径 ======
    'SOP': {
        'rmse_root': f"./{ROOT}/SOP-base-RMSE",                # TODO: SOP-RMSE 结果目录
        'mae_root': f"./{ROOT}/SOP-base-MAE",                  # TODO: SOP-MAE 结果目录
    }
}

# 3×5 baseline 总图的输出目录
FULL_OUTPUT_DIR = "./20260314_full_violin_figs"

# 所有“单独小提琴图”的统一输出目录
SINGLE_FIG_OUTPUT_DIR = "./20260314_single_violin_figs_baseline"

# transfer 情况下，SOC/SOP 的列名对应的横坐标显示格式
TEST_ID_TRANS = {
    'A_results': r'98\%-97\%',
    'B_results': r'97\%-96\%',
    'C_results': r'92\%-91\%',
    'D_results': r'91\%-90\%',
}


# transfer 的结果路径配置（根据你自己的实际目录改）
TRANSFER_CONFIG = {
    'SOH': {
        # 假设 SOH-transfer 的 xlsx 在这个目录下，
        # 文件名形如：CNN-MIT-Trans-UDDS-results.xlsx
        'root': f"./{ROOT}/SOH-transfer-Error",
        'mae_root': f"./{ROOT}/SOH-transfer",  # 不用也没关系，留着占位
    },
    'SOC': {
        # SOC transfer：RMSE/MAE 分别放在两个目录
        'rmse_root': f"./{ROOT}/SOC-transfer-RMSE",
        'mae_root': f"./{ROOT}/SOC-transfer-MAE",
    },
    'SOP': {
        # SOP transfer：RMSE/MAE 分别放在两个目录
        'rmse_root': f"./{ROOT}/SOP-transfer-RMSE",
        'mae_root': f"./{ROOT}/SOP-transfer-MAE",
    },
}

# transfer 单独小提琴图输出目录（避免和 baseline 混在一起）
TRANSFER_SINGLE_FIG_OUTPUT_DIR = "./20260314_single_violin_figs_transfer"



DISTILL_CONFIG = {
    'SOH': {
        # 假设 distill 的 SOH 在这个目录，文件名类似：
        # "Distilled student-UDDS-results.xlsx"
        'root': f"./{ROOT}/SOH-distill-Error",
    },
    'SOC': {
        # distill SOC：RMSE/MAE 分开存
        'rmse_root': f"./{ROOT}/SOC-distill-RMSE",
        'mae_root': f"./{ROOT}/SOC-distill-MAE",
    },
    'SOP': {
        'rmse_root': f"./{ROOT}/SOP-distill-RMSE",
        'mae_root': f"./{ROOT}/SOP-distill-MAE",
    },
}

DISTILL_SINGLE_FIG_OUTPUT_DIR = "./20260314_single_violin_figs_distill"



# ==============================
# 工具函数：读单个模型结果
# ==============================
def load_model_results(root_dir: str, model: str, dataset: str, test_id_s):
    """
    读取单个模型在单个数据集上的结果，并转换为统一格式：
    返回列包含 ['model', 'metric', 'error'] 的 DataFrame
    """
    possible_paths = [
        os.path.join(root_dir, f"{model}-{dataset}-results.xlsx"),
    ]

    df_raw = None
    for path in possible_paths:
        if os.path.exists(path):
            # 兼容两种 sheet 名
            for sheet in ['Sheet1', 'battery_mean_0']:
                try:
                    df_raw = pd.read_excel(path, engine='openpyxl', sheet_name=sheet)
                    break
                except Exception:
                    continue
        if df_raw is not None:
            break

    if df_raw is None:
        raise FileNotFoundError(f"No valid result file found for model={model}, dataset={dataset}, root={root_dir}")

    # 只保留 runs / experiment 1-10（如果存在）
    if 'runs' in df_raw.columns:
        df_raw = df_raw[df_raw['runs'].between(1, 10)]
    if 'experiment' in df_raw.columns:
        df_raw = df_raw[df_raw['experiment'].between(1, 10)]

    df_raw['model'] = model

    melted = pd.melt(
        df_raw,
        id_vars=['model'],
        value_vars=test_id_s,
        var_name='metric',
        value_name='error',
    )
    return melted


# ==============================
# baseline 工具函数：NASA / CS2 把 4 个电池/Cell 的结果取平均
# ==============================
def load_model_results_avg_over_test_ids(root_dir: str, model: str, dataset: str, test_id_s, metric_name: str):
    """
    baseline 专用：读取单个模型在单个数据集上的结果，将多个 test_id 列（如 4 个电池）做行均值，
    得到一列 error，并标注为 metric_name（MAE 或 RMSE）。

    返回列：['model', 'metric', 'error']
    """
    possible_paths = [
        os.path.join(root_dir, f"{model}-{dataset}-results.xlsx"),
    ]

    df_raw = None
    for path in possible_paths:
        if os.path.exists(path):
            for sheet in ['Sheet1', 'battery_mean_0']:
                try:
                    df_raw = pd.read_excel(path, engine='openpyxl', sheet_name=sheet)
                    break
                except Exception:
                    continue
        if df_raw is not None:
            break

    if df_raw is None:
        raise FileNotFoundError(f"No valid result file found for model={model}, dataset={dataset}, root={root_dir}")

    # 只保留 runs / experiment 1-10（如果存在）
    if 'runs' in df_raw.columns:
        df_raw = df_raw[df_raw['runs'].between(1, 10)]
    if 'experiment' in df_raw.columns:
        df_raw = df_raw[df_raw['experiment'].between(1, 10)]

    # 兼容列缺失：只对存在的列取均值
    existing_cols = [c for c in test_id_s if c in df_raw.columns]
    if len(existing_cols) == 0:
        raise KeyError(
            f"None of test_id columns found in file for model={model}, dataset={dataset}. "
            f"Expected one of: {test_id_s}"
        )

    df_raw['model'] = model
    # 行均值：把 B0005/B0006/B0007/B0018 或 CS2_35/36/37/38 平均成一个数
    df_raw['error'] = df_raw[existing_cols].mean(axis=1)

    out = df_raw[['model', 'error']].copy()
    out['metric'] = metric_name
    return out[['model', 'metric', 'error']]

# ==============================
# baseline v5：NASA / CS2 不再对 test_id 取平均，而是按电池/Cell 单独出图
# ==============================
def load_model_results_single_test_id(root_dir: str, model: str, dataset: str,
                                      test_id: str, metric_name: str):
    """
    baseline v5 专用：读取单个模型在单个数据集上的结果，只取一个 test_id 列（如 B0005 或 CS2_35），
    并标注为 metric_name（MAE 或 RMSE）。

    返回列：['model', 'metric', 'error']
    """
    possible_paths = [
        os.path.join(root_dir, f"{model}-{dataset}-results.xlsx"),
    ]

    df_raw = None
    for path in possible_paths:
        if os.path.exists(path):
            for sheet in ['Sheet1', 'battery_mean_0']:
                try:
                    df_raw = pd.read_excel(path, engine='openpyxl', sheet_name=sheet)
                    break
                except Exception:
                    continue
        if df_raw is not None:
            break

    if df_raw is None:
        raise FileNotFoundError(
            f"No valid result file found for model={model}, dataset={dataset}, root={root_dir}"
        )

    # 只保留 runs / experiment 1-10（如果存在）
    if 'runs' in df_raw.columns:
        df_raw = df_raw[df_raw['runs'].between(1, 10)]
    if 'experiment' in df_raw.columns:
        df_raw = df_raw[df_raw['experiment'].between(1, 10)]

    if test_id not in df_raw.columns:
        raise KeyError(
            f"Column '{test_id}' not found in file for model={model}, dataset={dataset}. "
            f"Available columns: {list(df_raw.columns)}"
        )

    out = pd.DataFrame({
        'model': model,
        'metric': metric_name,
        'error': df_raw[test_id].to_numpy(),
    })
    return out


def build_baseline_combined_mae_rmse_dataframe_per_test_id(
    rmse_root: str,
    mae_root: str,
    dataset: str,
    test_id: str,
):
    """
    baseline v5：对 NASA/CS2 的某一个 test_id（电池/Cell）单独构造一张 “MAE+RMSE” 合并图。

    返回：
      df_combined: 列 ['model', 'metric', 'error', 'metric_code']
      test_id_s: ['MAE', 'RMSE']
    """
    if dataset not in ['NASA', 'CS2']:
        raise ValueError(
            f"build_baseline_combined_mae_rmse_dataframe_per_test_id only supports NASA/CS2, got {dataset}"
        )

    df_list = []
    for model in METHODS:
        df_mae = load_model_results_single_test_id(
            root_dir=mae_root, model=model, dataset=dataset, test_id=test_id, metric_name='MAE'
        )
        df_rmse = load_model_results_single_test_id(
            root_dir=rmse_root, model=model, dataset=dataset, test_id=test_id, metric_name='RMSE'
        )
        df_list.extend([df_mae, df_rmse])

    df = pd.concat(df_list, axis=0).reset_index(drop=True)
    df['metric'] = pd.Categorical(df['metric'], categories=['MAE', 'RMSE'], ordered=True)
    df['metric_code'] = df['metric'].cat.codes
    return df, ['MAE', 'RMSE']




def build_baseline_combined_mae_rmse_dataframe(rmse_root: str, mae_root: str, dataset: str):
    """
    baseline 专用：NASA/CS2 的 MAE 与 RMSE 合并为同一张图（横轴只有 MAE、RMSE）。

    - RMSE：从 rmse_root 读取，取 4 个 test_id 的行均值
    - MAE：从 mae_root 读取，取 4 个 test_id 的行均值

    返回：
      df_combined: 列 ['model', 'metric', 'error', 'metric_code']
      test_id_s: ['MAE', 'RMSE']
    """
    test_id_s = DATA_TEST_IDS.get(dataset)
    if dataset not in ['NASA', 'CS2'] or test_id_s is None:
        raise ValueError(f"build_baseline_combined_mae_rmse_dataframe only supports NASA/CS2, got {dataset}")

    df_list = []
    for model in METHODS:
        df_rmse = load_model_results_avg_over_test_ids(rmse_root, model, dataset, test_id_s, metric_name='RMSE')
        df_mae  = load_model_results_avg_over_test_ids(mae_root,  model, dataset, test_id_s, metric_name='MAE')
        df_list.append(df_mae)
        df_list.append(df_rmse)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)
    # 强制 metric 顺序：MAE, RMSE
    df['metric'] = pd.Categorical(df['metric'], categories=['MAE', 'RMSE'], ordered=True)
    df['metric_code'] = df['metric'].cat.codes
    return df, ['MAE', 'RMSE']


def build_transfer_combined_mae_rmse_dataframe(rmse_root: str, mae_root: str, dataset: str, test_cols):
    """
    transfer 专用：把 SOC/SOP 的 MAE 与 RMSE 合并到同一张图（横轴只有 MAE、RMSE），并对 A/B/C/D 取行均值。
    返回 df_combined（列：model/metric/error/metric_code）与 test_id_s=['MAE','RMSE']
    """
    df_list = []
    for model in METHODS:
        df_rmse = load_model_results_avg_over_test_ids(
            rmse_root, model, dataset, test_cols, metric_name='RMSE'
        )
        df_mae = load_model_results_avg_over_test_ids(
            mae_root, model, dataset, test_cols, metric_name='MAE'
        )
        df_list.append(df_mae)
        df_list.append(df_rmse)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)
    df['metric'] = pd.Categorical(df['metric'], categories=['MAE', 'RMSE'], ordered=True)
    df['metric_code'] = df['metric'].cat.codes
    return df, ['MAE', 'RMSE']


def build_combined_mae_rmse_dataframe_single_test(rmse_root: str, mae_root: str, dataset: str, test_id: str):
    """
    通用：把某个任务（SOC / SOP）的 MAE 与 RMSE 合并到同一张图（横轴只有 MAE、RMSE），
    但只使用一个 test_id 列（如 A_results / B_results / ...），不做 A/B/C/D 平均。

    返回 df（列：model/metric/error/metric_code）与 test_id_s=['MAE','RMSE']
    """
    df_list = []
    for model in METHODS:
        df_mae = load_model_results_single_test_id(
            mae_root, model, dataset, test_id=test_id, metric_name='MAE'
        )
        df_rmse = load_model_results_single_test_id(
            rmse_root, model, dataset, test_id=test_id, metric_name='RMSE'
        )
        df_list.extend([df_mae, df_rmse])

    df = pd.concat(df_list, axis=0).reset_index(drop=True)
    df['metric'] = pd.Categorical(df['metric'], categories=['MAE', 'RMSE'], ordered=True)
    df['metric_code'] = df['metric'].cat.codes
    return df, ['MAE', 'RMSE']



def build_distill_combined_mae_rmse_dataframe(rmse_root: str, mae_root: str,
                                              dataset: str, test_cols):
    """
    distill 专用：SOC / SOP
    - A/B/C/D 取行均值
    - MAE + RMSE 合并
    """
    df_list = []
    for model in DISTILL_METHODS:  # 注意：这里是 DISTILL_METHODS
        df_mae = load_model_results_avg_over_test_ids(
            mae_root, model, dataset, test_cols, metric_name='MAE'
        )
        df_rmse = load_model_results_avg_over_test_ids(
            rmse_root, model, dataset, test_cols, metric_name='RMSE'
        )
        df_list.extend([df_mae, df_rmse])

    df = pd.concat(df_list, axis=0).reset_index(drop=True)
    df['metric'] = pd.Categorical(df['metric'], ['MAE', 'RMSE'], ordered=True)
    df['metric_code'] = df['metric'].cat.codes
    return df, ['MAE', 'RMSE']


# ==============================
# 工具函数：构造一个数据集的 DataFrame
# ==============================
def build_dataset_dataframe(root_dir: str, dataset: str):
    """
    对于一个数据集，读取所有模型结果，合并为一个 DataFrame：
    列：['model', 'metric', 'error', 'metric_code']
    """
    test_id_s = DATA_TEST_IDS.get(dataset, ['MAE', 'RMSE'])
    df_list = []
    for model in METHODS:
        df_model = load_model_results(root_dir, model, dataset, test_id_s)
        df_list.append(df_model)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)
    df['metric_code'] = df['metric'].astype('category').cat.codes
    return df, test_id_s


def build_transfer_soc_sop_dataframe(root_dir: str, dataset: str, test_cols):
    """
    transfer 情况下 SOC / SOP：
    - dataset 固定为 'MIT-Trans-UDDS'
    - test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    """
    df_list = []
    for model in METHODS:
        df_model = load_model_results(root_dir, model, dataset, test_cols)
        df_list.append(df_model)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)
    df['metric_code'] = df['metric'].astype('category').cat.codes
    return df



# ==============================
# 工具函数：在给定轴上画小提琴
# ==============================
def draw_violin(ax, df: pd.DataFrame):
    """
    在指定 ax 上绘制 violinplot。
    df 需包含列：['model', 'metric_code', 'error']
    """
    sns.violinplot(
        x='metric_code',
        y='error',
        hue='model',
        data=df,
        density_norm='count',
        inner='point',
        dodge=True,
        saturation=1,
        palette=COLORS,
        linewidth=0.6,
        ax=ax,
    )
    
    ax.get_legend().remove()


# ==============================
# 工具函数：画均值和标准差线
# ==============================
def draw_mean_std_lines(ax, df: pd.DataFrame, test_id_s):
    """
    在 ax 上为每个 metric + model 画均值和标准差线。
    """
    results_summary = []

    for i, metric_name in enumerate(test_id_s):
        for model in METHODS:
            subset = df[(df['model'] == model) & (df['metric_code'] == i)]
            if subset.empty:
                continue

            model_mean = subset['error'].mean()
            model_std = subset['error'].std()

            results_summary.append({
                "metric": metric_name,
                "model": model,
                "mean": model_mean,
                "std": model_std,
            })

            # 根据模型数量设置 x 方向偏移，使均值/方差线对齐对应的小提琴
            if len(METHODS) == 5:
                offset = 0.16
                x_pos = (
                    i
                    - (model == METHODS[0]) * offset * 2
                    - (model == METHODS[1]) * offset
                    + (model == METHODS[3]) * offset
                    + (model == METHODS[4]) * 2 * offset
                )
            elif len(METHODS) == 4:
                offset = 0.1
                x_pos = (
                    i
                    - (model == METHODS[0]) * offset * 3
                    - (model == METHODS[1]) * offset
                    + (model == METHODS[2]) * offset
                    + (model == METHODS[3]) * offset * 3
                )
            elif len(METHODS) == 3:
                # distill 场景：3 个 method
                offset = 0.27
                x_pos = i + (model == METHODS[2]) * offset - (model == METHODS[0]) * offset
            elif len(METHODS) == 6:
                offset = 0.0667
                x_pos = (
                    i
                    - (model == METHODS[0]) * offset * 5
                    - (model == METHODS[1]) * offset * 3
                    - (model == METHODS[2]) * offset
                    + (model == METHODS[3]) * offset
                    + (model == METHODS[4]) * offset * 3
                    + (model == METHODS[5]) * offset * 5
                )
            else:
                x_pos = i

            # 标准差竖线
            ax.plot(
                [x_pos, x_pos],
                [model_mean - model_std, model_mean + model_std],
                color='black',
                linestyle='-',
                linewidth=0.5,
            )
            # 均值横线
            ax.plot(
                [x_pos - 0.05, x_pos + 0.05],
                [model_mean, model_mean],
                color='red',
                linestyle='-',
                linewidth=0.6,
            )

    return results_summary

class ScalarFormatterOneDecimal(ScalarFormatter):
    """保持科学计数法 + 刻度标签强制一位小数"""

    def format_ticks(self, values):
        # 先用父类生成带科学计数法的刻度（含 mathtext）
        formatted = super().format_ticks(values)
        new_formatted = []
        for t in formatted:
            s = t

            # 去掉最外层的 $ ... $
            if s.startswith('$') and s.endswith('$'):
                s = s[1:-1]

            # 去掉 \mathdefault{...} 这层包裹
            if s.startswith(r'\mathdefault{') and s.endswith('}'):
                s = s[len(r'\mathdefault{'):-1]

            try:
                v = float(s)
                # 强制一位小数，比如 1 -> 1.0
                new_formatted.append(f"{v:.1f}")
            except ValueError:
                # 非数值（比如空字符串、别的标注）原样保留
                new_formatted.append(t)

        return new_formatted



def __old_format_axis(ax, dataset: str, test_id_s, ylabel: str,
                quantity: str, title_suffix: str = None,
                xlabels=None, mode: str = 'baseline'):
    """
    统一设置坐标轴、标题、字体等
    mode:
      - 'baseline': 原来的标题格式：DATASET: QUANTITY estimation
      - 'transfer': 新标题格式：QUANTITY estimation: Transfer learning
    """
    # ----- X 轴刻度与标签 -----
    labels = xlabels if xlabels is not None else test_id_s
    ax.set_xticks(range(len(test_id_s)))
    ax.set_xticklabels(labels, fontsize=FONT_SIZE)

    # Y 轴
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)

    # ----- 标题 -----
    if mode == 'baseline':
        base_title = DATA_TABLE.get(dataset, dataset)
        title = f"{base_title}: {quantity}"
    elif mode == 'transfer':
        # 只强调 quantity + transfer
        title = f"Transfer learning: {quantity} "
    elif mode == 'distill':
        # distill 的标题样式
        title = f"Knowledge-distillation: {quantity}"
    else:
        base_title = DATA_TABLE.get(dataset, dataset)
        title = f"{base_title}: {quantity} estimation"


    ax.set_xlabel(None)
    ax.set_title(title, fontsize=FONT_SIZE)

    # 边框线
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    sf = ScalarFormatterOneDecimal(useMathText=True)
    sf.set_scientific(True)
    sf.set_powerlimits((-1, 3))
    ax.yaxis.set_major_formatter(sf)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))



def format_axis(ax, dataset: str, test_id_s, ylabel: str,
                quantity: str, title_suffix: str = None,
                xlabels=None, mode: str = 'baseline'):
    """
    统一设置坐标轴、标题、字体等
    mode:
      - 'baseline': DATASET: QUANTITY
      - 'transfer': Transfer learning: QUANTITY
      - 'distill' : Knowledge-distillation: QUANTITY
    """

    # ----- X 轴刻度与标签 -----
    labels = xlabels if xlabels is not None else test_id_s
    ax.set_xticks(range(len(test_id_s)))
    ax.set_xticklabels(labels, fontsize=FONT_SIZE)

    # ----- Y 轴 -----
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)

    # ----- 标题 -----
    if mode == 'baseline':
        base_title = DATA_TABLE.get(dataset, dataset)
        title = f"{base_title}: {quantity}"
    elif mode == 'transfer':
        title = f"Transfer: {quantity}"
    elif mode == 'distill':
        title = f"KD: {quantity}"
    else:
        base_title = DATA_TABLE.get(dataset, dataset)
        title = f"{base_title}: {quantity}"

    # 可选：在标题中追加子标题（比如电池/Cell ID）
    if title_suffix is not None and str(title_suffix).strip() and str(title_suffix) != 'Error':
        suffix = str(title_suffix).strip()

        # # 标题太长时自动换行（更适合 600dpi 小图与多子图布局）
        # if ("\\n" in suffix) or (len(title) + len(suffix) + 3 > 42):
        #     title = f"{title}\\n({suffix})"
        # else:
        #     title = f"{title} ({suffix})"
        title = f"{title} ({suffix})"

    ax.set_xlabel(None)
    ax.set_title(title, fontsize=FONT_SIZE)
    

    # ----- 边框 -----
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # ======================================================
    # ✅ 关键修改：根据纵轴数值范围自动选择 formatter
    # ======================================================
    ymin, ymax = ax.get_ylim()
    scale = max(abs(ymin), abs(ymax))

    if scale < 3e-2:
        # --- 科学计数法 ---
        sf = ScalarFormatter(useMathText=True)
        sf.set_scientific(True)
        sf.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(sf)
    else:
        # --- 固定两位小数 ---
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

# ==============================
# 单独保存一张小提琴图（统一输出目录）
# ==============================
def save_single_violin_plot(
    df: pd.DataFrame,
    test_id_s,
    dataset: str,
    quantity: str,
    ylabel: str,
    filename: str,
    title_suffix: str,
    output_dir: str = SINGLE_FIG_OUTPUT_DIR,
    xlabels=None,
    mode: str = 'baseline',
):
    """
    为单个数据集另存一张单独的小提琴图（含均值/标准差线）
    baseline 与 transfer 复用，通过 mode 控制标题格式。
    """
    fig = plt.figure(figsize=(4, 3), dpi=600)
    ax = fig.add_subplot(111)

    draw_violin(ax, df)
    draw_mean_std_lines(ax, df, test_id_s)
    format_axis(ax,
                dataset,
                test_id_s,
                ylabel=ylabel,
                quantity=quantity,
                title_suffix=title_suffix,
                xlabels=xlabels,
                mode=mode)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")




def save_baseline_full_figure_3cols(baseline_config: dict, output_dir: str = FULL_OUTPUT_DIR):
    """
    baseline 总图（新版本）：共 9 张图，排成 3 行 × 3 列（不留空）。

    每个 quantity（SOH / SOC / SOP）对应一行，顺序为：
      [NASA(MAE+RMSE 合并), CS2(MAE+RMSE 合并), MIT(Error)]
    """
    quantities = ['SOH', 'SOC', 'SOP']
    num_rows, num_cols = 3, 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 9), dpi=600)

    for r, quantity in enumerate(quantities):
        cfg = baseline_config[quantity]
        rmse_root = cfg['rmse_root']
        mae_root  = cfg['mae_root']

        # --- NASA 合并图 ---
        ax = axes[r, 0]
        df_nasa, test_id_s = build_baseline_combined_mae_rmse_dataframe(rmse_root, mae_root, 'NASA')
        draw_violin(ax, df_nasa)
        draw_mean_std_lines(ax, df_nasa, test_id_s)
        format_axis(ax, dataset='NASA', test_id_s=test_id_s, ylabel='Error', quantity=quantity, title_suffix='Error')

        # --- CS2 合并图 ---
        ax = axes[r, 1]
        df_cs2, test_id_s = build_baseline_combined_mae_rmse_dataframe(rmse_root, mae_root, 'CS2')
        draw_violin(ax, df_cs2)
        draw_mean_std_lines(ax, df_cs2, test_id_s)
        format_axis(ax, dataset='CS2', test_id_s=test_id_s, ylabel='Error', quantity=quantity, title_suffix='Error')

        # --- MIT Error（MAE + RMSE 两列，保持原逻辑） ---
        ax = axes[r, 2]
        df_mit, test_id_s_mit = build_dataset_dataframe(rmse_root, 'MIT')
        draw_violin(ax, df_mit)
        draw_mean_std_lines(ax, df_mit, test_id_s_mit)
        format_axis(ax, dataset='MIT', test_id_s=test_id_s_mit, ylabel='Error', quantity=quantity, title_suffix='Error')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "Baseline_3x3_SOH_SOC_SOP.png")
    fig.savefig(save_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved baseline full figure (3x3): {save_path}")




# ==============================
# baseline v5：每个 quantity 生成 9 张图（NASA 4 + CS2 4 + MIT 1）的大图
# ==============================
def save_baseline_quantity_figure_3x3(
    baseline_config: dict,
    quantity: str,
    output_dir: str = FULL_OUTPUT_DIR,
):
    """
    baseline v5：单个 quantity（SOH / SOC / SOP）输出 3×3 的 9 子图大图：
      - NASA: B0005/B0006/B0007/B0018（每个电池一张，横轴 MAE/RMSE）
      - CS2 : CS2_35/CS2_36/CS2_37/CS2_38（每个 Cell 一张，横轴 MAE/RMSE）
      - MIT : 1 张（原逻辑，横轴 MAE/RMSE）
    """
    os.makedirs(output_dir, exist_ok=True)

    cfg = baseline_config[quantity]
    rmse_root = cfg['rmse_root']
    mae_root  = cfg['mae_root']

    panels = []
    # NASA 4
    for tid in DATA_TEST_IDS['NASA']:
        panels.append(('NASA', tid))
    # CS2 4
    for tid in DATA_TEST_IDS['CS2']:
        panels.append(('CS2', tid))
    # MIT 1（tid 为 None）
    panels.append(('MIT', None))

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), dpi=600)
    axes = axes.flatten()

    for idx, (dataset, tid) in enumerate(panels):
        ax = axes[idx]
        if dataset in ['NASA', 'CS2']:
            df, test_id_s = build_baseline_combined_mae_rmse_dataframe_per_test_id(
                rmse_root=rmse_root, mae_root=mae_root, dataset=dataset, test_id=tid
            )
            draw_violin(ax, df)
            draw_mean_std_lines(ax, df, test_id_s)
            format_axis(
                ax, dataset=dataset, test_id_s=test_id_s, ylabel='Error',
                quantity=quantity, title_suffix=tid, xlabels=['MAE', 'RMSE'], mode='baseline'
            )
        else:
            df_mit, test_id_s_mit = build_dataset_dataframe(rmse_root, 'MIT')
            draw_violin(ax, df_mit)
            draw_mean_std_lines(ax, df_mit, test_id_s_mit)
            format_axis(
                ax, dataset='MIT', test_id_s=test_id_s_mit, ylabel='Error',
                quantity=quantity, title_suffix='Error', mode='baseline'
            )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"Baseline_{quantity}_3x3_9plots.png")
    fig.savefig(out_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved baseline {quantity} figure (3x3): {out_path}")


# ==============================
# baseline v5：27 张图合并大图（SOH/SOC/SOP 共 27）
# ==============================
def save_baseline_full_figure_6x5_27plots(
    baseline_config: dict,
    output_dir: str = FULL_OUTPUT_DIR,
):
    """
    baseline v5：输出 27 张小提琴图合并后的大图。

    组合逻辑：
      - 每个 quantity（SOH/SOC/SOP）都有 9 张：
        NASA 4 + CS2 4 + MIT 1
      - 总共 27 张

    布局：
      - 默认 6×5（共 30 格），最后留 3 个空位。
    """
    os.makedirs(output_dir, exist_ok=True)

    panels = []
    for quantity in ['SOH', 'SOC', 'SOP']:
        # NASA 4
        for tid in DATA_TEST_IDS['NASA']:
            panels.append((quantity, 'NASA', tid))
        # CS2 4
        for tid in DATA_TEST_IDS['CS2']:
            panels.append((quantity, 'CS2', tid))
        # MIT 1
        panels.append((quantity, 'MIT', None))

    nrows, ncols = 6, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 18), dpi=600)
    axes = axes.flatten()

    for i in range(nrows * ncols):
        ax = axes[i]
        if i >= len(panels):
            ax.axis('off')
            continue

        quantity, dataset, tid = panels[i]
        cfg = baseline_config[quantity]
        rmse_root = cfg['rmse_root']
        mae_root  = cfg['mae_root']

        if dataset in ['NASA', 'CS2']:
            df, test_id_s = build_baseline_combined_mae_rmse_dataframe_per_test_id(
                rmse_root=rmse_root, mae_root=mae_root, dataset=dataset, test_id=tid
            )
            draw_violin(ax, df)
            draw_mean_std_lines(ax, df, test_id_s)
            format_axis(
                ax, dataset=dataset, test_id_s=test_id_s, ylabel='Error',
                quantity=quantity, title_suffix=tid, xlabels=['MAE', 'RMSE'], mode='baseline'
            )
        else:
            df_mit, test_id_s_mit = build_dataset_dataframe(rmse_root, 'MIT')
            draw_violin(ax, df_mit)
            draw_mean_std_lines(ax, df_mit, test_id_s_mit)
            format_axis(
                ax, dataset='MIT', test_id_s=test_id_s_mit, ylabel='Error',
                quantity=quantity, title_suffix='Error', mode='baseline'
            )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "Baseline_full_6x5_27plots.png")
    fig.savefig(out_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved baseline full figure (6x5, 27 plots): {out_path}")


def save_transfer_full_figure(transfer_config: dict, output_dir: str = "./full_violin_figs_transfer"):
    """
    transfer 总图（v5 更新）：
    - 3x3 共 9 张：
        1) SOH（Error）
        2) SOC：A/B/C/D 四组（MAE+RMSE 合并）
        3) SOP：A/B/C/D 四组（MAE+RMSE 合并）
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = 'MIT-Trans-UDDS'
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = ['MAE', 'RMSE']

    def _pretty_level(k: str) -> str:
        v = TEST_ID_TRANS.get(k, k)
        return v
        return str(v).replace('\%', '%')

    # 9 个子图顺序
    panels = [
        ('SOH', None),
        ('SOC', 'A_results'), ('SOC', 'B_results'),
        ('SOC', 'C_results'), ('SOC', 'D_results'),
        ('SOP', 'A_results'), ('SOP', 'B_results'),
        ('SOP', 'C_results'), ('SOP', 'D_results'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9.6), dpi=600)
    axes = axes.flatten()

    for i, (quantity, k) in enumerate(panels):
        ax = axes[i]

        if quantity == 'SOH':
            cfg = transfer_config['SOH']
            df, test_id_s = build_dataset_dataframe(cfg['root'], dataset)
            draw_violin(ax, df)
            draw_mean_std_lines(ax, df, test_id_s)
            format_axis(ax, dataset=dataset, test_id_s=test_id_s, ylabel='Error', quantity='SOH', title_suffix='Error',
                        xlabels=None, mode='transfer')
        else:
            cfg = transfer_config[quantity]
            df, test_id_s = build_combined_mae_rmse_dataframe_single_test(
                rmse_root=cfg['rmse_root'],
                mae_root=cfg['mae_root'],
                dataset=dataset,
                test_id=k
            )
            draw_violin(ax, df)
            draw_mean_std_lines(ax, df, test_id_s)
            format_axis(ax, dataset=dataset, test_id_s=test_id_s, ylabel='Error', quantity=quantity,
                        title_suffix=f"{_pretty_level(k)}", xlabels=xlabels, mode='transfer')

    plt.tight_layout()
    out_path = os.path.join(output_dir, "Transfer_full_figure_9plots_3x3.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)



def save_distill_full_figure(distill_config: dict, output_dir: str = "./full_violin_figs_distill"):
    """
    distill 总图（v5 更新）：
    - 3x3 共 9 张：
        1) SOH（Error）
        2) SOC：A/B/C/D 四组（MAE+RMSE 合并）
        3) SOP：A/B/C/D 四组（MAE+RMSE 合并）
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = 'UDDS'
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = ['MAE', 'RMSE']

    def _pretty_level(k: str) -> str:
        v = TEST_ID_TRANS.get(k, k)
        return v
        return str(v).replace('\%', '%')

    panels = [
        ('SOH', None),
        ('SOC', 'A_results'), ('SOC', 'B_results'),
        ('SOC', 'C_results'), ('SOC', 'D_results'),
        ('SOP', 'A_results'), ('SOP', 'B_results'),
        ('SOP', 'C_results'), ('SOP', 'D_results'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9.6), dpi=600)
    axes = axes.flatten()

    for i, (quantity, k) in enumerate(panels):
        ax = axes[i]
        if quantity == 'SOH':
            cfg = distill_config['SOH']
            df, test_id_s = build_dataset_dataframe(cfg['root'], dataset)
            draw_violin(ax, df)
            draw_mean_std_lines(ax, df, test_id_s)
            format_axis(ax, dataset=dataset, test_id_s=test_id_s, ylabel='Error', quantity='SOH', title_suffix='Error',
                        xlabels=None, mode='distill')
        else:
            cfg = distill_config[quantity]
            df, test_id_s = build_combined_mae_rmse_dataframe_single_test(
                rmse_root=cfg['rmse_root'],
                mae_root=cfg['mae_root'],
                dataset=dataset,
                test_id=k
            )
            draw_violin(ax, df)
            draw_mean_std_lines(ax, df, test_id_s)
            format_axis(ax, dataset=dataset, test_id_s=test_id_s, ylabel='Error', quantity=quantity,
                        title_suffix=f"{_pretty_level(k)}", xlabels=xlabels, mode='distill')

    plt.tight_layout()
    out_path = os.path.join(output_dir, "Distill_full_figure_9plots_3x3.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def run_baseline():
    """
    baseline v5 逻辑（不再对 NASA/CS2 的 4 个 test_id 取均值）：

    对每个 quantity（SOH / SOC / SOP）：
      - NASA: 4 个电池分别出图（每张图横轴 MAE/RMSE）
      - CS2 : 4 个 Cell 分别出图（每张图横轴 MAE/RMSE）
      - MIT : 仍为 1 张（原逻辑，横轴 MAE/RMSE）

    因此：
      - 单个 quantity 共有 9 张单图
      - 三个 quantity 共 27 张单图
      - 额外输出：
          1) 27 张合并大图（6×5 留空 3 格）
          2) SOH/SOC/SOP 各自 1 张 9 子图合并图（3×3）
    """
    os.makedirs(SINGLE_FIG_OUTPUT_DIR, exist_ok=True)

    for quantity, cfg in BASELINE_CONFIG.items():
        rmse_root = cfg['rmse_root']
        mae_root  = cfg['mae_root']

        # ---- NASA / CS2：每个 test_id 单独画（MAE+RMSE 合并）----
        for dataset in ['NASA', 'CS2']:
            for test_id in DATA_TEST_IDS[dataset]:
                df_combined, test_id_s = build_baseline_combined_mae_rmse_dataframe_per_test_id(
                    rmse_root=rmse_root, mae_root=mae_root, dataset=dataset, test_id=test_id
                )
                filename = f"{quantity}-{dataset}-{test_id}-Error.png"
                save_single_violin_plot(
                    df=df_combined,
                    test_id_s=test_id_s,
                    dataset=dataset,
                    quantity=quantity,
                    ylabel="Error",
                    filename=filename,
                    title_suffix=test_id,  # 标题里追加电池/Cell ID
                    output_dir=SINGLE_FIG_OUTPUT_DIR,
                    xlabels=['MAE', 'RMSE'],
                    mode='baseline',
                )

        # ---- MIT：保持原逻辑 ----
        df_mit, test_id_s_mit = build_dataset_dataframe(rmse_root, 'MIT')
        filename = f"{quantity}-MIT-Error.png"
        save_single_violin_plot(
            df=df_mit,
            test_id_s=test_id_s_mit,
            dataset='MIT',
            quantity=quantity,
            ylabel="Error",
            filename=filename,
            title_suffix="Error",
            output_dir=SINGLE_FIG_OUTPUT_DIR,
            mode='baseline',
        )

        # ---- 额外：该 quantity 的 9 子图合并图 ----
        save_baseline_quantity_figure_3x3(BASELINE_CONFIG, quantity=quantity, output_dir=FULL_OUTPUT_DIR)

    # ---- 额外：27 子图合并大图 ----
    save_baseline_full_figure_6x5_27plots(BASELINE_CONFIG, output_dir=FULL_OUTPUT_DIR)





def run_transfer():
    """
    transfer v5（更新逻辑）：
    - SOH：保持原逻辑（MIT-Trans-UDDS 的 Error，1 张）
    - SOC：A/B/C/D 四组分别画（每组 1 张，横轴 MAE、RMSE）
    - SOP：A/B/C/D 四组分别画（每组 1 张，横轴 MAE、RMSE）

    因此 transfer 总共会输出 1 + 4 + 4 = 9 张单图，并生成 1 张 3x3 的合并总图。
    """
    os.makedirs(TRANSFER_SINGLE_FIG_OUTPUT_DIR, exist_ok=True)

    dataset = 'MIT-Trans-UDDS'
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = ['MAE', 'RMSE']

    def _pretty_level(k: str) -> str:
        # TEST_ID_TRANS 里用了 latex 风格的 \% ，这里展示用普通 %
        v = TEST_ID_TRANS.get(k, k)
        return v
        # return str(v).replace('\%', '%')

    # ========= SOH：单图 =========
    soh_cfg = TRANSFER_CONFIG['SOH']
    df_soh, test_id_s = build_dataset_dataframe(soh_cfg['root'], dataset)
    save_single_violin_plot(
        df=df_soh,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOH',
        ylabel='Error',
        filename="SOH-MIT-Trans-UDDS-Error.png",
        title_suffix="Error",
        output_dir=TRANSFER_SINGLE_FIG_OUTPUT_DIR,
        xlabels=None,
        mode='transfer',
    )

    # ========= SOC：A-D 分开画（MAE+RMSE 合并，不取均值）=========
    soc_cfg = TRANSFER_CONFIG['SOC']
    for k in test_cols:
        df_soc, test_id_s = build_combined_mae_rmse_dataframe_single_test(
            rmse_root=soc_cfg['rmse_root'],
            mae_root=soc_cfg['mae_root'],
            dataset=dataset,
            test_id=k
        )
        save_single_violin_plot(
            df=df_soc,
            test_id_s=test_id_s,
            dataset=dataset,
            quantity='SOC',
            ylabel='Error',
            filename=f"SOC-{k}-MIT-Trans-UDDS-Error.png",
            title_suffix=f"{_pretty_level(k)}",
            output_dir=TRANSFER_SINGLE_FIG_OUTPUT_DIR,
            xlabels=xlabels,
            mode='transfer',
        )

    # ========= SOP：A-D 分开画（MAE+RMSE 合并，不取均值）=========
    sop_cfg = TRANSFER_CONFIG['SOP']
    for k in test_cols:
        df_sop, test_id_s = build_combined_mae_rmse_dataframe_single_test(
            rmse_root=sop_cfg['rmse_root'],
            mae_root=sop_cfg['mae_root'],
            dataset=dataset,
            test_id=k
        )
        save_single_violin_plot(
            df=df_sop,
            test_id_s=test_id_s,
            dataset=dataset,
            quantity='SOP',
            ylabel='Error',
            filename=f"SOP-{k}-MIT-Trans-UDDS-Error.png",
            title_suffix=f"{_pretty_level(k)}",
            output_dir=TRANSFER_SINGLE_FIG_OUTPUT_DIR,
            xlabels=xlabels,
            mode='transfer',
        )

    # ========= 生成整图（9 张）=========
    save_transfer_full_figure(TRANSFER_CONFIG, output_dir=FULL_OUTPUT_DIR)


def __old_run_distill():
    """
    distill 情况：
    - SOH：1 张（dataset='UDDS'，默认列 ['MAE', 'RMSE']）
    - SOC：2 张，RMSE/MAE，列为 A_results/B_results/C_results/D_results
    - SOP：2 张，同 SOC
    """
    # distill 专用的 3 个方法

    os.makedirs(DISTILL_SINGLE_FIG_OUTPUT_DIR, exist_ok=True)

    # ========= SOH：单图 =========
    soh_cfg = DISTILL_CONFIG['SOH']
    soh_root = soh_cfg['root']
    # 假设 SOH distill 使用 dataset='UDDS'，列名 ['MAE','RMSE']
    df_soh, test_id_s_soh = build_dataset_dataframe(soh_root, 'UDDS')
    save_single_violin_plot(
        df=df_soh,
        test_id_s=test_id_s_soh,
        dataset='UDDS',
        quantity='SOH',
        ylabel='Error',
        filename="SOH-UDDS-Error-distill.png",
        title_suffix="Error",
        output_dir=DISTILL_SINGLE_FIG_OUTPUT_DIR,
        mode='distill',
    )

    # ========= SOC：RMSE / MAE 单图 =========
    soc_cfg = DISTILL_CONFIG['SOC']
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = [TEST_ID_TRANS[col] for col in test_cols]

    # SOC RMSE
    df_soc_rmse = build_transfer_soc_sop_dataframe(
        soc_cfg['rmse_root'], dataset='UDDS', test_cols=test_cols
    )
    save_single_violin_plot(
        df=df_soc_rmse,
        test_id_s=test_cols,
        dataset='UDDS',
        quantity='SOC',
        ylabel='RMSE',
        filename="SOC-UDDS-RMSE-distill.png",
        title_suffix="RMSE",
        output_dir=DISTILL_SINGLE_FIG_OUTPUT_DIR,
        xlabels=xlabels,
        mode='distill',
    )

    # SOC MAE
    df_soc_mae = build_transfer_soc_sop_dataframe(
        soc_cfg['mae_root'], dataset='UDDS', test_cols=test_cols
    )
    save_single_violin_plot(
        df=df_soc_mae,
        test_id_s=test_cols,
        dataset='UDDS',
        quantity='SOC',
        ylabel='MAE',
        filename="SOC-UDDS-MAE-distill.png",
        title_suffix="MAE",
        output_dir=DISTILL_SINGLE_FIG_OUTPUT_DIR,
        xlabels=xlabels,
        mode='distill',
    )

    # ========= SOP：RMSE / MAE 单图 =========
    sop_cfg = DISTILL_CONFIG['SOP']

    # SOP RMSE
    df_sop_rmse = build_transfer_soc_sop_dataframe(
        sop_cfg['rmse_root'], dataset='UDDS', test_cols=test_cols
    )
    save_single_violin_plot(
        df=df_sop_rmse,
        test_id_s=test_cols,
        dataset='UDDS',
        quantity='SOP',
        ylabel='RMSE',
        filename="SOP-UDDS-RMSE-distill.png",
        title_suffix="RMSE",
        output_dir=DISTILL_SINGLE_FIG_OUTPUT_DIR,
        xlabels=xlabels,
        mode='distill',
    )

    # SOP MAE
    df_sop_mae = build_transfer_soc_sop_dataframe(
        sop_cfg['mae_root'], dataset='UDDS', test_cols=test_cols
    )
    save_single_violin_plot(
        df=df_sop_mae,
        test_id_s=test_cols,
        dataset='UDDS',
        quantity='SOP',
        ylabel='MAE',
        filename="SOP-UDDS-MAE-distill.png",
        title_suffix="MAE",
        output_dir=DISTILL_SINGLE_FIG_OUTPUT_DIR,
        xlabels=xlabels,
        mode='distill',
    )

    # ========= 可选：也画一张 distill 总图，布局和 transfer 一样 =========
    save_distill_full_figure(DISTILL_CONFIG, FULL_OUTPUT_DIR)





def run_distill():
    """
    distill v5（更新逻辑，参考 transfer）：
    - SOH：1 张
    - SOC：A/B/C/D 四组分别画（每组 1 张，横轴 MAE、RMSE）
    - SOP：A/B/C/D 四组分别画（每组 1 张，横轴 MAE、RMSE）

    distill 总共输出 1 + 4 + 4 = 9 张单图，并生成 1 张 3x3 的合并总图。
    """
    os.makedirs(DISTILL_SINGLE_FIG_OUTPUT_DIR, exist_ok=True)

    dataset = 'UDDS'
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = ['MAE', 'RMSE']

    def _pretty_level(k: str) -> str:
        v = TEST_ID_TRANS.get(k, k)
        return v 
        return str(v).replace('\%', '%')

    # ========= SOH =========
    soh_cfg = DISTILL_CONFIG['SOH']
    df_soh, test_id_s = build_dataset_dataframe(soh_cfg['root'], dataset)
    save_single_violin_plot(
        df=df_soh,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOH',
        ylabel='Error',
        filename="SOH-UDDS-Error.png",
        title_suffix="Error",
        output_dir=DISTILL_SINGLE_FIG_OUTPUT_DIR,
        xlabels=None,
        mode='distill',
    )

    # ========= SOC：A-D 分开（MAE+RMSE 合并，不取均值）=========
    soc_cfg = DISTILL_CONFIG['SOC']
    for k in test_cols:
        df_soc, test_id_s = build_combined_mae_rmse_dataframe_single_test(
            rmse_root=soc_cfg['rmse_root'],
            mae_root=soc_cfg['mae_root'],
            dataset=dataset,
            test_id=k
        )
        save_single_violin_plot(
            df=df_soc,
            test_id_s=test_id_s,
            dataset=dataset,
            quantity='SOC',
            ylabel='Error',
            filename=f"SOC-{k}-UDDS-Error.png",
            title_suffix=f"{_pretty_level(k)}",
            output_dir=DISTILL_SINGLE_FIG_OUTPUT_DIR,
            xlabels=xlabels,
            mode='distill',
        )

    # ========= SOP：A-D 分开（MAE+RMSE 合并，不取均值）=========
    sop_cfg = DISTILL_CONFIG['SOP']
    for k in test_cols:
        df_sop, test_id_s = build_combined_mae_rmse_dataframe_single_test(
            rmse_root=sop_cfg['rmse_root'],
            mae_root=sop_cfg['mae_root'],
            dataset=dataset,
            test_id=k
        )
        save_single_violin_plot(
            df=df_sop,
            test_id_s=test_id_s,
            dataset=dataset,
            quantity='SOP',
            ylabel='Error',
            filename=f"SOP-{k}-UDDS-Error.png",
            title_suffix=f"{_pretty_level(k)}",
            output_dir=DISTILL_SINGLE_FIG_OUTPUT_DIR,
            xlabels=xlabels,
            mode='distill',
        )

    # ========= 生成整图（9 张）=========
    save_distill_full_figure(DISTILL_CONFIG, output_dir=FULL_OUTPUT_DIR)


if __name__ == "__main__":
    if TASK == 'baseline':
        run_baseline()
    elif TASK == 'transfer':
        run_transfer()
    elif TASK == 'distill':
        run_distill()
    else:
        raise ValueError(f"Unknown TASK type: {TASK}")

