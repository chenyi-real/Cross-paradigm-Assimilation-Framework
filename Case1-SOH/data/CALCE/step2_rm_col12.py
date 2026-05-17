# remove_first_two_cols_csv.py
# -*- coding: utf-8 -*-
"""
批量删除 CSV 文件前两列，并保存到 new_out 目录。

用法：
python remove_first_two_cols_csv.py --in_dir ./out --out_dir ./new_out
"""

import sys
import argparse
from pathlib import Path
import pandas as pd


def remove_first_two_columns(csv_path: Path, out_dir: Path):
    """删除 CSV 第一和第二列，并保存"""
    """删除 CSV 第十五列，并保存"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[读取失败] {csv_path.name}: {e}", file=sys.stderr)
        return None

    if df.shape[1] <= 2:
        print(f"[提示] {csv_path.name}: 列数 ≤ 2，无法删除前两列，已跳过。")
        return None

    # 删除前两列（按位置）
    df_new = df.drop(df.columns[[0, 1, 15]], axis=1)

    # 保存到 new_out
    out_path = out_dir / csv_path.name
    try:
        df_new.to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[保存失败] {csv_path.name}: {e}", file=sys.stderr)
        return None

    return out_path


def main():
    ap = argparse.ArgumentParser(description="批量删除 CSV 前两列")
    ap.add_argument("--in_dir", type=Path, required=True, help="输入目录（包含 CSV 文件）")
    ap.add_argument("--out_dir", type=Path, required=True, help="输出目录（new_out）")
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.csv"))
    if not files:
        print(f"[提示] 输入目录中未找到 .csv 文件：{in_dir}")
        sys.exit(0)

    print(f"发现 {len(files)} 个 CSV 文件，将输出到 {out_dir}")
    for f in files:
        print(f"处理：{f.name}")
        out_path = remove_first_two_columns(f, out_dir)
        if out_path:
            print(f"  -> 保存到: {out_path}")


if __name__ == "__main__":
    main()
