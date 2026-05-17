#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 Excel 表格的“第一列”拆分为 B0005/B0006/B0007/B0018 四个 CSV 文件。

用法示例：
  python split_to_csv.py -i HIs_temp.xlsx
  python split_to_csv.py -i HIs_temp.xlsx -o out_csv
  python split_to_csv.py -i HIs_temp.xlsx --column Battery  # 指定列名（否则默认第一列）
  python split_to_csv.py -i HIs_temp.xlsx --skip-empty      # 不导出空文件
"""

import argparse
import os
import sys
import pandas as pd

TARGET_IDS = ["B0005", "B0006", "B0007", "B0018"]

def read_table(path: str) -> pd.DataFrame:
    """读取 Excel/CSV（自动识别后缀）"""
    try:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        # .xlsx 用 openpyxl 更稳；.xls 可省略 engine 让 pandas 自行处理（需额外依赖）
        return pd.read_excel(path, engine="openpyxl" if path.lower().endswith(".xlsx") else None)
    except Exception as e:
        print(f"[错误] 无法读取文件：{path}\n{e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="按第一列拆分 Excel 为 4 个 CSV：B0005/B0006/B0007/B0018")
    parser.add_argument("-i", "--input", required=True, help="输入表格路径（.xlsx/.xls/.csv）")
    parser.add_argument("-o", "--output-dir", default="out", help="输出目录（默认：out）")
    parser.add_argument("--column", help="用于分组的列名；不填则使用第一列")
    parser.add_argument("--exact", action="store_true",
                        help="严格匹配单元格值（默认宽松匹配：转字符串并去空格）")
    parser.add_argument("--encoding", default="utf-8-sig",
                        help="CSV 编码（默认：utf-8-sig，便于 Excel 打开）")
    parser.add_argument("--skip-empty", action="store_true", help="不导出空文件")
    args = parser.parse_args()

    df = read_table(args.input)
    if df.empty:
        print("[提示] 输入表格无数据。")
        sys.exit(0)

    # 确定用于分组的列
    if args.column:
        if args.column not in df.columns:
            print(f"[错误] 未找到列：{args.column}。可选列：{list(df.columns)}")
            sys.exit(1)
        key_series = df[args.column]
        key_col_name = args.column
    else:
        key_series = df.iloc[:, 0]
        key_col_name = df.columns[0]

    # 宽松/严格匹配
    if args.exact:
        keys = key_series
    else:
        keys = key_series.astype(str).str.strip()

    os.makedirs(args.output_dir, exist_ok=True)

    # 导出
    print(f"[信息] 使用列：{key_col_name}；目标ID：{', '.join(TARGET_IDS)}")
    total_exported = 0
    for tid in TARGET_IDS:
        mask = keys.eq(tid)
        sub = df.loc[mask]
        if args.skip_empty and sub.empty:
            print(f"[跳过] {tid}: 0 行（--skip-empty 生效）")
            continue
        out_path = os.path.join(args.output_dir, f"{tid}.csv")
        try:
            sub.to_csv(out_path, index=False, encoding=args.encoding)
            print(f"[完成] {tid}: {len(sub)} 行 -> {out_path}")
            total_exported += 1
        except Exception as e:
            print(f"[错误] 写入 {out_path} 失败：{e}")

    # 统计
    unmatched = ~keys.isin(TARGET_IDS)
    print("\n[统计]")
    for tid in TARGET_IDS:
        print(f"  {tid}: {(keys == tid).sum()} 行")
    print(f"  其他/未匹配: {int(unmatched.sum())} 行（未导出）")

    if total_exported == 0:
        print("\n[提示] 未导出任何文件，请检查目标 ID 与第一列数据是否一致，或尝试去掉 --exact。")

if __name__ == "__main__":
    main()
