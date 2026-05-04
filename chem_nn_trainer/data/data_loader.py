# ┌────────────────────────────────────────────────────────────┐
# │  CSV 数据加载器                                            │
# │  负责从 CSV 文件加载数据并提供统计信息                      │
# └────────────────────────────────────────────────────────────┘

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class DataLoader:
    """CSV 数据加载器

    功能：
    - 从指定路径加载 CSV 文件为 pandas DataFrame
    - 缓存统计结果避免重复计算
    - 提供每列的基础统计信息（均值、标准差、最小/最大值、缺失值等）
    """

    def __init__(self):
        self.df = None              # 当前加载的 DataFrame
        self.file_path = None       # 当前文件路径
        self._stats_cache = None    # 统计信息缓存，避免重复计算

    def load_csv(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """加载 CSV 文件并返回数据框与元信息

        Args:
            file_path: CSV 文件的绝对路径

        Returns:
            (DataFrame, dict): 数据框和包含行数、列数、列名等信息的字典
        """
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self._stats_cache = None    # 清空缓存，新数据需重新计算

        data_info = {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "file_path": file_path
        }
        return self.df, data_info

    def get_all_stats(self) -> Dict:
        """获取所有列的统计信息（带缓存）

        对每个数值列计算：缺失数、缺失率、均值、标准差、最小值、最大值。
        非数值列只报告缺失信息，统计指标记为 None。

        Returns:
            dict: 以列名为键，统计字典为值的嵌套字典
        """
        if self.df is None:
            return {}

        # 命中缓存则直接返回
        if self._stats_cache is not None:
            return self._stats_cache

        stats = {}
        for col in self.df.columns:
            missing = int(self.df[col].isna().sum())
            missing_rate = (missing / len(self.df)) * 100 if len(self.df) > 0 else 0

            # 尝试将列转为数值类型，非数值会变为 NaN
            numeric_data = pd.to_numeric(self.df[col], errors='coerce')

            if numeric_data.notna().sum() > 0:
                stats[col] = {
                    "missing": missing,
                    "missing_rate": round(missing_rate, 1),
                    "mean": round(float(numeric_data.mean()), 2),
                    "std": round(float(numeric_data.std()), 2),
                    "min": round(float(numeric_data.min()), 2),
                    "max": round(float(numeric_data.max()), 2),
                    "dtype": str(self.df[col].dtype)
                }
            else:
                # 非数值列：统计指标置为 None
                stats[col] = {
                    "missing": missing,
                    "missing_rate": round(missing_rate, 1),
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "dtype": str(self.df[col].dtype)
                }

        self._stats_cache = stats
        return stats
