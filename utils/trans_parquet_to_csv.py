# coding=utf-8
# utils/trans_parquet_to_csv.py

"""
Parquet 文件转换为 CSV 文件的模块

该模块负责将指定目录下的所有 *.parquet 文件转换为同名的 *.csv 文件。
转换完成后的 CSV 文件将保存在相同的目录中。

用法：
    python trans_parquet_to_csv.py
"""

import os
import pandas as pd
from pathlib import Path


def convert_parquet_to_csv(parquet_file_path):
    """
    将单个 Parquet 文件转换为 CSV 文件。

    参数：
        parquet_file_path (Path): 要转换的 Parquet 文件路径。

    返回：
        csv_file_path (Path): 转换后的 CSV 文件路径，若转换失败则为 None。
    """
    try:
        # 读取 Parquet 文件
        df = pd.read_parquet(parquet_file_path)

        # 构建 CSV 文件路径，保持相同的文件名，不改变扩展名
        csv_file_path = parquet_file_path.with_suffix('.csv')

        # 将 DataFrame 保存为 CSV 文件，不包含索引
        df.to_csv(csv_file_path, index=False, encoding='utf-8')

        print(f"成功转换：'{parquet_file_path.name}' -> '{csv_file_path.name}'")
        return csv_file_path
    except Exception as e:
        print(f"转换文件 '{parquet_file_path.name}' 时出错。错误信息：{e}")
        return None


def batch_convert_dataset(dataset_dir):
    """
    批量将指定目录下的所有 Parquet 文件转换为 CSV 文件。

    参数：
        dataset_dir (str or Path): 数据集所在的目录路径。

    返回：
        None
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"错误：目录 '{dataset_dir}' 不存在。")
        return

    # 获取所有 .parquet 文件
    parquet_files = list(dataset_path.glob("*.parquet"))

    if not parquet_files:
        print(f"警告：在目录 '{dataset_dir}' 中未找到任何 .parquet 文件。")
        return

    print(f"找到 {len(parquet_files)} 个 .parquet 文件，开始转换...")

    for parquet_file in parquet_files:
        convert_parquet_to_csv(parquet_file)

    print("所有转换操作已完成。")


if __name__ == "__main__":
    # 定义数据集目录，相对于当前脚本的位置
    dataset_directory = Path(__file__).parent.parent / "dataset"

    # 执行批量转换
    batch_convert_dataset(dataset_directory)
