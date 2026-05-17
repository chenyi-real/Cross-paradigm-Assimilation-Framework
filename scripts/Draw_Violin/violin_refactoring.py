'''
当前v4的版本不是用来分batch的, 是把所有batch一起算的平均.
20251221版本就是靠这个v4画出来的

增加一个ablation模式
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
# 任务类型：'baseline' 或 'transfer' 或 'distill' 或 'ablation' 或 'validity' 或 'ablation_baseline'
# 其中 'ablation_baseline' 是专门用来画 baseline 的 ablation 图的（只画 Full / WO_AAOs / WO_Mechanism / WO_Perception 四个方法）
# 其中validity 是专门用来画公测数据集上有效性图的（只画 AAOs / NW / DW / PA 四个方法）
# ==============================
TASK = 'validity_udds'  # 改成 'transfer' 即可运行 transfer 任务


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
BASELINE_METHODS = ['Ours', 'MLP', 'CNN', 'LSTM', 'TCN', "PINN"]
DISTILL_METHODS = ['Distilled student', 'Teacher', 'Non-distilled student']
ABLATION_METHODS = ['Full', 'WO_Perception', 'WO_Mechanism', 'WO_AAOs']
VALIDITY_METHODS = ['AAOs', 'NW', 'DW', 'PA']
# 默认先用 baseline 的方法（baseline / transfer 共用）

if TASK =='distill':
    METHODS = DISTILL_METHODS
elif TASK == 'ablation' or TASK == 'ablation_baseline':
    METHODS = ABLATION_METHODS
elif TASK == 'validity' or TASK == 'validity_udds':
    METHODS = VALIDITY_METHODS
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


ROOT = './20260313_exp_data_sum'
# ROOT = './20260125_ablation'

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
# FULL_OUTPUT_DIR = "./20251221_violin_results/full_violin_figs"
FULL_OUTPUT_DIR = "./20260313_violin_results/full_violin_figs"

# 所有“单独小提琴图”的统一输出目录
SINGLE_FIG_OUTPUT_DIR = "./20260313_violin_results/single_violin_figs_baseline"


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
        'rmse_root': f"./{ROOT}/SOH-transfer-Error",
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
TRANSFER_SINGLE_FIG_OUTPUT_DIR = "./20260313_violin_results/single_violin_figs_transfer"



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

DISTILL_SINGLE_FIG_OUTPUT_DIR = "./20260313_violin_results/single_violin_figs_distill"




ABLATION_CONFIG = {
    'SOH': {
        # 假设 distill 的 SOH 在这个目录，文件名类似：
        # "Distilled student-UDDS-results.xlsx"
        'root': f"./{ROOT}/SOH-ablation-Error",
    },
    'SOC': {
        # distill SOC：RMSE/MAE 分开存
        'rmse_root': f"./{ROOT}/SOC-ablation-RMSE",
        'mae_root': f"./{ROOT}/SOC-ablation-MAE",
    },
    'SOP': {
        'rmse_root': f"./{ROOT}/SOP-ablation-RMSE",
        'mae_root': f"./{ROOT}/SOP-ablation-MAE",
    },
}

ABLATION_SINGLE_FIG_OUTPUT_DIR = "./20260313_violin_results/single_violin_figs_ablation"


ABLATION_BASELINE_CONFIG = {
    # ====== 已经有的 SOH 配置（沿用你原来的路径） ======
    'SOH': {
        'rmse_root': f"./{ROOT}/SOH-ablation-RMSE",        # SOH-RMSE 结果目录
        'mae_root': f"./{ROOT}/SOH-ablation-MAE",     # SOH-MAE 结果目录
    },

    # ====== 你填上 SOC 的结果路径 ======
    'SOC': {
        'rmse_root': f"./{ROOT}/SOC-ablation-RMSE",                # TODO: SOC-RMSE 结果目录
        'mae_root': f"./{ROOT}/SOC-ablation-MAE",                  # TODO: SOC-MAE 结果目录
    },

    # ====== 你填上 SOP 的结果路径 ======
    'SOP': {
        'rmse_root': f"./{ROOT}/SOP-ablation-RMSE",                # TODO: SOP-RMSE 结果目录
        'mae_root': f"./{ROOT}/SOP-ablation-MAE",                  # TODO: SOP-MAE 结果目录
    }
}

ABLATION_BASELINE_SINGLE_FIG_OUTPUT_DIR = "./20260313_violin_results/single_violin_figs_ablation"


VALIDITY_CONFIG = {
    # ====== 已经有的 SOH 配置（沿用你原来的路径） ======
    'SOH': {
        'rmse_root': f"./{ROOT}/SOH-validity-RMSE",        # SOH-RMSE 结果目录
        'mae_root': f"./{ROOT}/SOH-validity-MAE",     # SOH-MAE 结果目录
    },

    # ====== 你填上 SOC 的结果路径 ======
    'SOC': {
        'rmse_root': f"./{ROOT}/SOC-validity-RMSE",                # TODO: SOC-RMSE 结果目录
        'mae_root': f"./{ROOT}/SOC-validity-MAE",                  # TODO: SOC-MAE 结果目录
    },

    # ====== 你填上 SOP 的结果路径 ======
    'SOP': {
        'rmse_root': f"./{ROOT}/SOP-validity-RMSE",                # TODO: SOP-RMSE 结果目录
        'mae_root': f"./{ROOT}/SOP-validity-MAE",                  # TODO: SOP-MAE 结果目录
    }
}

VALIDITY_SINGLE_FIG_OUTPUT_DIR = "./20260313_violin_results/single_violin_figs_validity"


VALIDITY_UDDS_CONFIG = {
    # ====== 已经有的 SOH 配置（沿用你原来的路径） ======
    'SOH': {
        # 假设 distill 的 SOH 在这个目录，文件名类似：
        # "Distilled student-UDDS-results.xlsx"
        'root': f"./{ROOT}/SOH-validity-UDDS-Error",
    },

    # ====== 你填上 SOC 的结果路径 ======
    'SOC': {
        'rmse_root': f"./{ROOT}/SOC-validity-UDDS-RMSE",                # TODO: SOC-RMSE 结果目录
        'mae_root': f"./{ROOT}/SOC-validity-UDDS-MAE",                  # TODO: SOC-MAE 结果目录
    },

    # ====== 你填上 SOP 的结果路径 ======
    'SOP': {
        'rmse_root': f"./{ROOT}/SOP-validity-UDDS-RMSE",                # TODO: SOP-RMSE 结果目录
        'mae_root': f"./{ROOT}/SOP-validity-UDDS-MAE",                  # TODO: SOP-MAE 结果目录
    }
}

VALIDITY_UDDS_SINGLE_FIG_OUTPUT_DIR = "./20260313_violin_results/single_violin_figs_validity_udds"




# ==============================
# 工具函数：读单个模型结果
# ==============================
def load_model_results(root_dir: str, model: str, dataset: str, test_id_s):
    """
    读取单个模型在单个数据集上的结果，并转换为统一格式：
    返回列包含 ['model', 'metric', 'error'] 的 DataFrame
    """
    possible_paths = [
        os.path.join(root_dir, f"{model}-{dataset}-results_adjusted.xlsx"),
        os.path.join(root_dir, f"{model}-{dataset}-results.xlsx"),
        os.path.join(root_dir, f"{model}_adjusted.xlsx"),
        os.path.join(root_dir, f"{model}.xlsx"),
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
        os.path.join(root_dir, f"{model}-{dataset}-results_adjusted.xlsx"),
        os.path.join(root_dir, f"{model}-{dataset}-results.xlsx"),
        os.path.join(root_dir, f"{model}_adjusted.xlsx"),
        os.path.join(root_dir, f"{model}.xlsx"),
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



def build_ablation_combined_mae_rmse_dataframe(rmse_root: str, mae_root: str,
                                              dataset: str, test_cols):
    """
    ablation 专用：SOC / SOP
    - A/B/C/D 取行均值
    - MAE + RMSE 合并
    """
    df_list = []
    for model in ABLATION_METHODS:  # 注意：这里是 ABLATION_METHODS
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



def build_validity_combined_mae_rmse_dataframe(rmse_root: str, mae_root: str,
                                              dataset: str, test_cols):
    """
    validity 专用：SOC / SOP
    - A/B/C/D 取行均值
    - MAE + RMSE 合并
    """
    df_list = []
    for model in VALIDITY_METHODS:  # 注意：这里是 VALIDITY_METHODS
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
                quantity: str, title_suffix: str = None, # type: ignore
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
                quantity: str, title_suffix: str = None, # type: ignore
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
        title = f"Transfer learning: {quantity}"
    elif mode == 'distill':
        title = f"Knowledge-distillation: {quantity}"
    elif mode == 'ablation':
        title = f'Ablation Study: {quantity}'
    elif mode == 'validity':
        title = f'Validity Study: {quantity}'
    else:
        base_title = DATA_TABLE.get(dataset, dataset)
        title = f"{base_title}: {quantity}"

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



def save_transfer_full_figure(transfer_config: dict, output_dir: str = "./full_violin_figs_transfer"):
    """
    transfer 总图（新版本）：
    布局：1 行 3 列
      [SOH-Error, SOC(MAE+RMSE 合并), SOP(MAE+RMSE 合并)]
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = 'MIT-Trans-UDDS'
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = ['MAE', 'RMSE']

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=600)

    # -------- SOH --------
    ax = axes[0]
    cfg = transfer_config['SOH']
    df_soh, test_id_s_soh = build_dataset_dataframe(cfg['rmse_root'], dataset)
    draw_violin(ax, df_soh)
    draw_mean_std_lines(ax, df_soh, test_id_s_soh)
    format_axis(ax, dataset=dataset, test_id_s=test_id_s_soh, ylabel='Error', quantity='SOH', title_suffix='Error', mode='transfer')

    # -------- SOC（合并 MAE+RMSE）--------
    ax = axes[1]
    cfg = transfer_config['SOC']
    df_soc, test_id_s = build_transfer_combined_mae_rmse_dataframe(
        rmse_root=cfg['rmse_root'],
        mae_root=cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    draw_violin(ax, df_soc)
    draw_mean_std_lines(ax, df_soc, test_id_s)
    format_axis(ax, dataset=dataset, test_id_s=test_id_s, ylabel='Error', quantity='SOC', title_suffix='Error',
                xlabels=xlabels, mode='transfer')

    # -------- SOP（合并 MAE+RMSE）--------
    ax = axes[2]
    cfg = transfer_config['SOP']
    df_sop, test_id_s = build_transfer_combined_mae_rmse_dataframe(
        rmse_root=cfg['rmse_root'],
        mae_root=cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    draw_violin(ax, df_sop)
    draw_mean_std_lines(ax, df_sop, test_id_s)
    format_axis(ax, dataset=dataset, test_id_s=test_id_s, ylabel='Error', quantity='SOP', title_suffix='Error',
                xlabels=xlabels, mode='transfer')

    plt.tight_layout()

    out_path = os.path.join(output_dir, "Transfer_full_figure_3plots.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)





def save_distill_full_figure(distill_config: dict,
                             output_dir: str = "./full_violin_figs_distill"):
    """
    distill 总图（新）：
    1 × 3 布局
      [SOH, SOC(MAE+RMSE), SOP(MAE+RMSE)]
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = 'UDDS'
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = ['MAE', 'RMSE']

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=600)

    # -------- SOH --------
    ax = axes[0]
    cfg = distill_config['SOH']
    df_soh, test_id_s = build_dataset_dataframe(cfg['root'], dataset)
    draw_violin(ax, df_soh)
    draw_mean_std_lines(ax, df_soh, test_id_s)
    format_axis(ax, dataset, test_id_s, 'Error', 'SOH', mode='distill')

    # -------- SOC --------
    ax = axes[1]
    cfg = distill_config['SOC']
    df_soc, test_id_s = build_distill_combined_mae_rmse_dataframe(
        rmse_root=cfg['rmse_root'],
        mae_root=cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    draw_violin(ax, df_soc)
    draw_mean_std_lines(ax, df_soc, test_id_s)
    format_axis(ax, dataset, test_id_s, 'Error', 'SOC',
                xlabels=xlabels, mode='distill')

    # -------- SOP --------
    ax = axes[2]
    cfg = distill_config['SOP']
    df_sop, test_id_s = build_distill_combined_mae_rmse_dataframe(
        rmse_root=cfg['rmse_root'],
        mae_root=cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    draw_violin(ax, df_sop)
    draw_mean_std_lines(ax, df_sop, test_id_s)
    format_axis(ax, dataset, test_id_s, 'Error', 'SOP',
                xlabels=xlabels, mode='distill')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "Distill_full_figure_3plots.png"),
                bbox_inches='tight')
    plt.close(fig)



def save_ablation_full_figure(ablation_config: dict,
                             output_dir: str = "./full_violin_figs_ablation"):
    """
    ablation 总图（新）：
    1 × 3 布局
      [SOH, SOC(MAE+RMSE), SOP(MAE+RMSE)]
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = 'UDDS'
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = ['MAE', 'RMSE']

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=600)

    # -------- SOH --------
    ax = axes[0]
    cfg = ablation_config['SOH']
    df_soh, test_id_s = build_dataset_dataframe(cfg['root'], dataset)
    draw_violin(ax, df_soh)
    draw_mean_std_lines(ax, df_soh, test_id_s)
    format_axis(ax, dataset, test_id_s, 'Error', 'SOH', mode='ablation')
    # -------- SOC --------
    ax = axes[1]
    cfg = ablation_config['SOC']
    df_soc, test_id_s = build_ablation_combined_mae_rmse_dataframe(
        rmse_root=cfg['rmse_root'],
        mae_root=cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    draw_violin(ax, df_soc)
    draw_mean_std_lines(ax, df_soc, test_id_s)
    format_axis(ax, dataset, test_id_s, 'Error', 'SOC',
                xlabels=xlabels, mode='ablation')

    # -------- SOP --------
    ax = axes[2]
    cfg = ablation_config['SOP']
    df_sop, test_id_s = build_ablation_combined_mae_rmse_dataframe(
        rmse_root=cfg['rmse_root'],
        mae_root=cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    draw_violin(ax, df_sop)
    draw_mean_std_lines(ax, df_sop, test_id_s)
    format_axis(ax, dataset, test_id_s, 'Error', 'SOP',
                xlabels=xlabels, mode='ablation')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "Ablation_full_figure_3plots.png"),
                bbox_inches='tight')
    plt.close(fig)



def save_validity_udds_full_figure_3cols(validity_config: dict,
                                         output_dir: str = "./full_violin_figs_validity_udds"):
    """
    validity_udds 总图（新）：
    1 × 3 布局
      [SOH, SOC(MAE+RMSE), SOP(MAE+RMSE)]
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = 'MIT-Trans-UDDS'
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = ['MAE', 'RMSE']

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=600)

    # -------- SOH --------
    ax = axes[0]
    cfg = validity_config['SOH']
    df_soh, test_id_s = build_dataset_dataframe(cfg['root'], dataset)
    draw_violin(ax, df_soh)
    draw_mean_std_lines(ax, df_soh, test_id_s)
    format_axis(ax, dataset, test_id_s, 'Error', 'SOH', mode='validity')
    # -------- SOC --------
    ax = axes[1]
    cfg = validity_config['SOC']
    df_soc, test_id_s = build_validity_combined_mae_rmse_dataframe(
        rmse_root=cfg['rmse_root'],
        mae_root=cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    draw_violin(ax, df_soc)
    draw_mean_std_lines(ax, df_soc, test_id_s)
    format_axis(ax, dataset, test_id_s, 'Error', 'SOC',
                xlabels=xlabels, mode='validity')

    # -------- SOP --------
    ax = axes[2]
    cfg = validity_config['SOP']
    df_sop, test_id_s = build_validity_combined_mae_rmse_dataframe(
        rmse_root=cfg['rmse_root'],
        mae_root=cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    draw_violin(ax, df_sop)
    draw_mean_std_lines(ax, df_sop, test_id_s)
    format_axis(ax, dataset, test_id_s, 'Error', 'SOP',
                xlabels=xlabels, mode='validity')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "Validity_full_figure_3plots.png"),
                bbox_inches='tight')
    plt.close(fig)





def save_ablation_baseline_full_figure_3cols(baseline_config: dict, output_dir: str = FULL_OUTPUT_DIR):
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
    save_path = os.path.join(output_dir, "Ablation_Baseline_3x3_SOH_SOC_SOP.png")
    fig.savefig(save_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved baseline full figure (3x3): {save_path}")


def save_validity_full_figure_3cols(baseline_config: dict, output_dir: str = FULL_OUTPUT_DIR):
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
    save_path = os.path.join(output_dir, "Validity_3x3_SOH_SOC_SOP.png")
    fig.savefig(save_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved validity full figure (3x3): {save_path}")

# ==============================
# 主流程：baseline 任务
# ==============================

def run_baseline():
    # baseline / transfer 共用 5-method 配置

    # 1) 统一目录下保存所有单独的小提琴图
    os.makedirs(SINGLE_FIG_OUTPUT_DIR, exist_ok=True)

    for quantity, cfg in BASELINE_CONFIG.items():
        rmse_root = cfg['rmse_root']
        mae_root = cfg['mae_root']

        # NASA & CS2：把 MAE+RMSE 合并到同一张图（横轴：MAE, RMSE）
        for dataset in ['NASA', 'CS2']:
            df_combined, test_id_s = build_baseline_combined_mae_rmse_dataframe(
                rmse_root=rmse_root, mae_root=mae_root, dataset=dataset
            )
            filename = f"{quantity}-{dataset}-Error.png"
            save_single_violin_plot(
                df=df_combined,
                test_id_s=test_id_s,
                dataset=dataset,
                quantity=quantity,
                ylabel="Error",
                filename=filename,
                title_suffix="Error",
                output_dir=SINGLE_FIG_OUTPUT_DIR,
                xlabels=['MAE', 'RMSE'],
            )

        # MIT: Error（MAE + RMSE 两列）保持不变
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
        )

    # 2) 生成 baseline 总图：3×3 布局，SOH -> SOC -> SOP
    save_baseline_full_figure_3cols(BASELINE_CONFIG, FULL_OUTPUT_DIR)





def run_transfer():
    """
    transfer 情况（新逻辑）：
    - 只画 3 张：
      1) SOH：1 张（MIT-Trans-UDDS 的 Error，保持原逻辑）
      2) SOC：1 张（MAE+RMSE 合并，A/B/C/D 取均值，横轴 MAE、RMSE）
      3) SOP：1 张（MAE+RMSE 合并，A/B/C/D 取均值，横轴 MAE、RMSE）
    """
    os.makedirs(TRANSFER_SINGLE_FIG_OUTPUT_DIR, exist_ok=True)

    dataset = 'MIT-Trans-UDDS'
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = ['MAE', 'RMSE']

    # ========= SOH：单图 =========
    soh_cfg = TRANSFER_CONFIG['SOH']
    df_soh, test_id_s_soh = build_dataset_dataframe(soh_cfg['rmse_root'], dataset)
    save_single_violin_plot(
        df=df_soh,
        test_id_s=test_id_s_soh,
        dataset=dataset,
        quantity='SOH',
        ylabel='Error',
        filename="SOH-MIT-Trans-UDDS-Error.png",
        title_suffix="Error",
        output_dir=TRANSFER_SINGLE_FIG_OUTPUT_DIR,
        xlabels=None,
        mode='transfer',
    )

    # ========= SOC：合并图（MAE+RMSE，A-D 取均值）=========
    soc_cfg = TRANSFER_CONFIG['SOC']
    df_soc, test_id_s = build_transfer_combined_mae_rmse_dataframe(
        rmse_root=soc_cfg['rmse_root'],
        mae_root=soc_cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    save_single_violin_plot(
        df=df_soc,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOC',
        ylabel='Error',
        filename="SOC-MIT-Trans-UDDS-Error.png",
        title_suffix="Error",
        output_dir=TRANSFER_SINGLE_FIG_OUTPUT_DIR,
        xlabels=xlabels,
        mode='transfer',
    )

    # ========= SOP：合并图（MAE+RMSE，A-D 取均值）=========
    sop_cfg = TRANSFER_CONFIG['SOP']
    df_sop, test_id_s = build_transfer_combined_mae_rmse_dataframe(
        rmse_root=sop_cfg['rmse_root'],
        mae_root=sop_cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    save_single_violin_plot(
        df=df_sop,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOP',
        ylabel='Error',
        filename="SOP-MIT-Trans-UDDS-Error.png",
        title_suffix="Error",
        output_dir=TRANSFER_SINGLE_FIG_OUTPUT_DIR,
        xlabels=xlabels,
        mode='transfer',
    )

    # ========= 生成整图（3 张）=========
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
    distill 新逻辑（只画 3 张）：
    - SOH：1 张
    - SOC：MAE + RMSE 合并（A/B/C/D 均值）
    - SOP：MAE + RMSE 合并（A/B/C/D 均值）
    """
    os.makedirs(DISTILL_SINGLE_FIG_OUTPUT_DIR, exist_ok=True)

    dataset = 'UDDS'
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = ['MAE', 'RMSE']

    # ========= SOH =========
    soh_cfg = DISTILL_CONFIG['SOH']
    df_soh, test_id_s = build_dataset_dataframe(soh_cfg['root'], dataset)
    save_single_violin_plot(
        df=df_soh,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOH',
        ylabel='Error',
        filename="SOH-MIT-Distill-Error.png",
        title_suffix="Error",
        output_dir=DISTILL_SINGLE_FIG_OUTPUT_DIR,
        mode='distill',
    )

    # ========= SOC =========
    soc_cfg = DISTILL_CONFIG['SOC']
    df_soc, test_id_s = build_distill_combined_mae_rmse_dataframe(
        rmse_root=soc_cfg['rmse_root'],
        mae_root=soc_cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    save_single_violin_plot(
        df=df_soc,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOC',
        ylabel='Error',
        filename="SOC-MIT-Distill-Error.png",
        title_suffix="Error",
        output_dir=DISTILL_SINGLE_FIG_OUTPUT_DIR,
        xlabels=xlabels,
        mode='distill',
    )

    # ========= SOP =========
    sop_cfg = DISTILL_CONFIG['SOP']
    df_sop, test_id_s = build_distill_combined_mae_rmse_dataframe(
        rmse_root=sop_cfg['rmse_root'],
        mae_root=sop_cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    save_single_violin_plot(
        df=df_sop,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOP',
        ylabel='Error',
        filename="SOP-MIT-Distill-Error.png",
        title_suffix="Error",
        output_dir=DISTILL_SINGLE_FIG_OUTPUT_DIR,
        xlabels=xlabels,
        mode='distill',
    )

    # ========= 总图 =========
    save_distill_full_figure(DISTILL_CONFIG, output_dir=FULL_OUTPUT_DIR)



def run_ablation_study():
    """
    distill 新逻辑（只画 3 张）：
    - SOH：1 张
    - SOC：MAE + RMSE 合并（A/B/C/D 均值）
    - SOP：MAE + RMSE 合并（A/B/C/D 均值）
    """
    os.makedirs(ABLATION_SINGLE_FIG_OUTPUT_DIR, exist_ok=True)

    dataset = 'UDDS'
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = ['MAE', 'RMSE']

    # ========= SOH =========
    soh_cfg = ABLATION_CONFIG['SOH']
    df_soh, test_id_s = build_dataset_dataframe(soh_cfg['root'], dataset)
    save_single_violin_plot(
        df=df_soh,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOH',
        ylabel='Error',
        filename="SOH-MIT-Ablation-Error.png",
        title_suffix="Error",
        output_dir=ABLATION_SINGLE_FIG_OUTPUT_DIR,
        mode='ablation',
    )

    # ========= SOC =========
    soc_cfg = ABLATION_CONFIG['SOC']
    df_soc, test_id_s = build_ablation_combined_mae_rmse_dataframe(
        rmse_root=soc_cfg['rmse_root'],
        mae_root=soc_cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    save_single_violin_plot(
        df=df_soc,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOC',
        ylabel='Error',
        filename="SOC-MIT-Ablation-Error.png",
        title_suffix="Error",
        output_dir=ABLATION_SINGLE_FIG_OUTPUT_DIR,
        xlabels=xlabels,
        mode='ablation',
    )

    # ========= SOP =========
    sop_cfg = ABLATION_CONFIG['SOP']
    df_sop, test_id_s = build_ablation_combined_mae_rmse_dataframe(
        rmse_root=sop_cfg['rmse_root'],
        mae_root=sop_cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    save_single_violin_plot(
        df=df_sop,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOP',
        ylabel='Error',
        filename="SOP-MIT-Ablation-Error.png",
        title_suffix="Error",
        output_dir=ABLATION_SINGLE_FIG_OUTPUT_DIR,
        xlabels=xlabels,
        mode='ablation',
    )

    # ========= 总图 =========
    save_ablation_full_figure(ABLATION_CONFIG, output_dir=FULL_OUTPUT_DIR)




def run_validity_analysis_udds():
    """
    validity_udds 新逻辑（只画 3 张）：
    - SOH：1 张
    - SOC：MAE + RMSE 合并（A/B/C/D 均值）
    - SOP：MAE + RMSE 合并（A/B/C/D 均值）
    """
    os.makedirs(VALIDITY_UDDS_SINGLE_FIG_OUTPUT_DIR, exist_ok=True)

    dataset = 'MIT-Trans-UDDS'
    test_cols = ['A_results', 'B_results', 'C_results', 'D_results']
    xlabels = ['MAE', 'RMSE']

    # ========= SOH =========
    soh_cfg = VALIDITY_UDDS_CONFIG['SOH']
    df_soh, test_id_s = build_dataset_dataframe(soh_cfg['root'], dataset)
    save_single_violin_plot(
        df=df_soh,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOH',
        ylabel='Error',
        filename="SOH-validity-udds-Error.png",
        title_suffix="Error",
        output_dir=VALIDITY_UDDS_SINGLE_FIG_OUTPUT_DIR,
        mode='validity',
    )

    # ========= SOC =========
    soc_cfg = VALIDITY_UDDS_CONFIG['SOC']
    df_soc, test_id_s = build_validity_combined_mae_rmse_dataframe(
        rmse_root=soc_cfg['rmse_root'],
        mae_root=soc_cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    save_single_violin_plot(
        df=df_soc,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOC',
        ylabel='Error',
        filename="SOC-validity-udds-Error.png",
        title_suffix="Error",
        output_dir=VALIDITY_UDDS_SINGLE_FIG_OUTPUT_DIR,
        xlabels=xlabels,
        mode='validity',
    )

    # ========= SOP =========
    sop_cfg = VALIDITY_UDDS_CONFIG['SOP']
    df_sop, test_id_s = build_validity_combined_mae_rmse_dataframe(
        rmse_root=sop_cfg['rmse_root'],
        mae_root=sop_cfg['mae_root'],
        dataset=dataset,
        test_cols=test_cols
    )
    save_single_violin_plot(
        df=df_sop,
        test_id_s=test_id_s,
        dataset=dataset,
        quantity='SOP',
        ylabel='Error',
        filename="SOP-validity-udds-Error.png",
        title_suffix="Error",
        output_dir=VALIDITY_UDDS_SINGLE_FIG_OUTPUT_DIR,
        xlabels=xlabels,
        mode='validity',
    )

    # ========= 总图 =========
    save_validity_udds_full_figure_3cols(VALIDITY_UDDS_CONFIG, output_dir=FULL_OUTPUT_DIR)



def run_ablation_baseline():

    # 1) 统一目录下保存所有单独的小提琴图
    os.makedirs(ABLATION_BASELINE_SINGLE_FIG_OUTPUT_DIR, exist_ok=True)

    for quantity, cfg in ABLATION_BASELINE_CONFIG.items():
        rmse_root = cfg['rmse_root']
        mae_root = cfg['mae_root']

        # NASA & CS2：把 MAE+RMSE 合并到同一张图（横轴：MAE, RMSE）
        for dataset in ['NASA', 'CS2']:
            df_combined, test_id_s = build_baseline_combined_mae_rmse_dataframe(
                rmse_root=rmse_root, mae_root=mae_root, dataset=dataset
            )
            filename = f"{quantity}-{dataset}-ablation-Error.png"
            save_single_violin_plot(
                df=df_combined,
                test_id_s=test_id_s,
                dataset=dataset,
                quantity=quantity,
                ylabel="Error",
                filename=filename,
                title_suffix="Error",
                output_dir=ABLATION_BASELINE_SINGLE_FIG_OUTPUT_DIR,
                xlabels=['MAE', 'RMSE'],
            )

        # MIT: Error（MAE + RMSE 两列）保持不变
        df_mit, test_id_s_mit = build_dataset_dataframe(rmse_root, 'MIT')
        filename = f"{quantity}-MIT-ablation-Error.png"
        save_single_violin_plot(
            df=df_mit,
            test_id_s=test_id_s_mit,
            dataset='MIT',
            quantity=quantity,
            ylabel="Error",
            filename=filename,
            title_suffix="Error",
            output_dir=ABLATION_BASELINE_SINGLE_FIG_OUTPUT_DIR,
        )

    # 2) 生成 baseline 总图：3×3 布局，SOH -> SOC -> SOP
    save_ablation_baseline_full_figure_3cols(ABLATION_BASELINE_CONFIG, FULL_OUTPUT_DIR)



def run_validity_analysis():

    # 1) 统一目录下保存所有单独的小提琴图
    os.makedirs(VALIDITY_SINGLE_FIG_OUTPUT_DIR, exist_ok=True)

    for quantity, cfg in VALIDITY_CONFIG.items():
        rmse_root = cfg['rmse_root']
        mae_root = cfg['mae_root']

        # NASA & CS2：把 MAE+RMSE 合并到同一张图（横轴：MAE, RMSE）
        for dataset in ['NASA', 'CS2']:
            df_combined, test_id_s = build_baseline_combined_mae_rmse_dataframe(
                rmse_root=rmse_root, mae_root=mae_root, dataset=dataset
            )
            filename = f"{quantity}-{dataset}-validity-Error.png"
            save_single_violin_plot(
                df=df_combined,
                test_id_s=test_id_s,
                dataset=dataset,
                quantity=quantity,
                ylabel="Error",
                filename=filename,
                title_suffix="Error",
                output_dir=VALIDITY_SINGLE_FIG_OUTPUT_DIR,
                xlabels=['MAE', 'RMSE'],
            )

        # MIT: Error（MAE + RMSE 两列）保持不变
        df_mit, test_id_s_mit = build_dataset_dataframe(rmse_root, 'MIT')
        filename = f"{quantity}-MIT-validity-Error.png"
        save_single_violin_plot(
            df=df_mit,
            test_id_s=test_id_s_mit,
            dataset='MIT',
            quantity=quantity,
            ylabel="Error",
            filename=filename,
            title_suffix="Error",
            output_dir=VALIDITY_SINGLE_FIG_OUTPUT_DIR,
        )

    # 2) 生成 validity 总图：3×3 布局，SOH -> SOC -> SOP
    save_validity_full_figure_3cols(VALIDITY_CONFIG, FULL_OUTPUT_DIR)


if __name__ == "__main__":
    if TASK == 'baseline':
        run_baseline()
    elif TASK == 'transfer':
        run_transfer()
    elif TASK == 'distill':
        run_distill()
    elif TASK == 'ablation':
        run_ablation_study()
    elif TASK == 'ablation_baseline':
        run_ablation_baseline()
    elif TASK == "validity":
        run_validity_analysis()
    elif TASK == "validity_udds":
        run_validity_analysis_udds()
    else:
        raise ValueError(f"Unknown TASK type: {TASK}")
