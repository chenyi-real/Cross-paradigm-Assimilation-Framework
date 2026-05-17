# batch_process_cs2.py
# -*- coding: utf-8 -*-
"""
批量处理：
- 对输入目录下所有 .xlsx 文件
- 读取第1个工作表
- 自动识别包含 "CS2_36_MM_DD_YY" 的列并解析日期（YY -> 2000+YY，不要求两位数）
- 先按解析日期升序，再按 Cycle 升序（二级排序）
- 导出排序后的 CSV 到 out 目录
- 基于排序后的顺序绘制 SoH 折线图，保存到 out 目录
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def parse_cs2_date(s: str):
    """解析 'CS2_36_MM_DD_YY'，月日年可为一位或两位数字，YY -> 2000+YY"""
    parts = str(s).split("_")
    if len(parts) < 5:
        return pd.NaT
    try:
        mm, dd, yy = int(parts[2]), int(parts[3]), int(parts[4])
        year = 2000 + yy
        return datetime(year, mm, dd)
    except Exception:
        return pd.NaT


def find_id_col(df: pd.DataFrame) -> str:
    """找到包含 'CS2_36_' 模式的列"""
    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            if df[col].dropna().astype(str).str.startswith("CS2_").any():
                return col
    raise ValueError("未找到符合 'CS2_36_' 模式的列。")


def find_soh_col(df: pd.DataFrame) -> Optional[str]:
    """寻找 SoH 列（不区分大小写）"""
    for col in df.columns:
        if str(col).lower() == "soh":
            return col
    for col in df.columns:
        if "soh" in str(col).lower():
            return col
    return None


def find_cycle_col(df: pd.DataFrame) -> Optional[str]:
    """寻找 Cycle 列（不区分大小写）"""
    for col in df.columns:
        if str(col).lower() == "cycle":
            return col
    for col in df.columns:
        if "cycle" in str(col).lower():
            return col
    return None


def to_numeric_safe(s: pd.Series) -> pd.Series:
    """安全转为数值"""
    return pd.to_numeric(s, errors="coerce")


def process_one_excel(xlsx_path: Path, out_dir: Path):
    """处理单个 Excel 文件"""
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0)
    except Exception as e:
        print(f"[读取失败] {xlsx_path.name}: {e}", file=sys.stderr)
        return None, None

    try:
        id_col = find_id_col(df)
    except Exception as e:
        print(f"[ID列未找到] {xlsx_path.name}: {e}", file=sys.stderr)
        return None, None

    df["_parsed_date"] = df[id_col].apply(parse_cs2_date)

    # 二级排序：Cycle
    cycle_col = find_cycle_col(df)
    if cycle_col:
        df["_cycle_numeric_tmp"] = to_numeric_safe(df[cycle_col])
        sort_by = ["_parsed_date", "_cycle_numeric_tmp"]
        ascending = [True, True]
        df_sorted = df.sort_values(by=sort_by, ascending=ascending, na_position="last").reset_index(drop=True)
        df_sorted.drop(columns=["_cycle_numeric_tmp"], inplace=True)
    else:
        df_sorted = df.sort_values(by="_parsed_date", ascending=True, na_position="last").reset_index(drop=True)

    # 保存 CSV
    csv_path = out_dir / f"{xlsx_path.stem}_sorted.csv"
    try:
        df_sorted.to_csv(csv_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[保存CSV失败] {xlsx_path.name}: {e}", file=sys.stderr)
        csv_path = None

    # 绘制 SoH 折线图
    soh_col = find_soh_col(df_sorted)
    plot_path = None
    if soh_col:
        try:
            plt.figure()
            plt.plot(df_sorted[soh_col].values)
            plt.title(f"SoH 折线图 - {xlsx_path.stem}")
            plt.xlabel("索引（按日期→Cycle 升序）" if cycle_col else "索引（按日期升序）")
            plt.ylabel(str(soh_col))
            plt.tight_layout()
            plot_path = out_dir / f"{xlsx_path.stem}_SoH.png"
            plt.savefig(plot_path, dpi=150)
        except Exception as e:
            print(f"[绘图失败] {xlsx_path.name}: {e}", file=sys.stderr)
        finally:
            plt.close()
    else:
        print(f"[提示] {xlsx_path.name}: 未找到 SoH 列，跳过绘图。", file=sys.stderr)

    return csv_path, plot_path


def main():
    ap = argparse.ArgumentParser(description="按日期和Cycle排序，并导出CSV和SoH折线图")
    ap.add_argument("--in_dir", type=Path, required=True, help="输入目录（包含 .xlsx 文件）")
    ap.add_argument("--out_dir", type=Path, required=True, help="输出目录")
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.xlsx"))
    if not files:
        print(f"[提示] 输入目录中未找到 .xlsx 文件：{in_dir}")
        sys.exit(0)

    print(f"发现 {len(files)} 个文件，将输出到 {out_dir}")
    for f in files:
        print(f"处理：{f.name}")
        csv_path, plot_path = process_one_excel(f, out_dir)
        if csv_path:
            print(f"  -> CSV: {csv_path}")
        if plot_path:
            print(f"  -> 图:  {plot_path}")


if __name__ == "__main__":
    main()
