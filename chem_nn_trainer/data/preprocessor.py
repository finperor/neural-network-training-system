# ┌────────────────────────────────────────────────────────────┐
# │  数据预处理器                                              │
# │  提供缺失值填充、异常值检测与处理、数据归一化等功能          │
# └────────────────────────────────────────────────────────────┘

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.impute import KNNImputer


class DataPreprocessor:
    """数据预处理器

    功能：
    - 自动分析数据质量（缺失率、建议处理方法、是否存在异常值）
    - 多种缺失值处理策略（均值/中位数/前向/后向/插值/KNN/自动选择）
    - 异常值检测（IQR 或 Z-Score）与处理（删除或裁剪）
    - 数据归一化（Min-Max 归一化或 Z-Score 标准化）
    """

    def __init__(self):
        self.norm_params = {}       # 归一化参数，用于后续反归一化

    def auto_analyze(self, df: pd.DataFrame) -> Dict:
        """自动分析数据质量

        根据缺失率高低和列名特征推荐合适的缺失值处理方法，
        同时使用 IQR 方法检测各数值列是否存在异常值。

        Args:
            df: 原始数据 DataFrame

        Returns:
            dict: 每列的分析结果，包含 missing_count, missing_rate,
                  suggested_missing_method, has_outliers
        """
        analysis = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_rate = (missing_count / len(df)) * 100 if len(df) > 0 else 0

            # 根据缺失率推荐处理方法
            if missing_rate < 5:
                suggested_method = "均值"
            elif missing_rate < 20:
                suggested_method = "KNN填充"
            elif "时间" in col or "time" in col.lower() or "date" in col.lower():
                suggested_method = "前向填充"
            else:
                suggested_method = "插值"

            # IQR 方法检测异常值
            has_outliers = False
            if col in numeric_cols and df[col].notna().sum() > 3:
                valid = df[col].dropna()
                if len(valid) > 0:
                    q1 = valid.quantile(0.25)
                    q3 = valid.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        outlier_count = ((valid < lower) | (valid > upper)).sum()
                        has_outliers = outlier_count > 0

            analysis[col] = {
                "missing_count": missing_count,
                "missing_rate": round(missing_rate, 1),
                "suggested_missing_method": suggested_method,
                "has_outliers": has_outliers
            }

        return analysis

    def handle_missing_values(self, df: pd.DataFrame, method: str,
                              auto_analysis: Optional[Dict] = None) -> pd.DataFrame:
        """处理缺失值

        支持方法：
        - "mean": 均值填充（数值列）/ 众数填充（非数值列）
        - "median": 中位数填充
        - "forward_fill": 前向填充（用前一有效值填充）
        - "backward_fill": 后向填充（用后一有效值填充）
        - "interpolate": 线性插值
        - "knn": KNN 填充（使用 sklearn.KNNImputer）
        - "auto": 根据 auto_analysis 结果自动选择

        Args:
            df: 含缺失值的数据
            method: 处理方法名称
            auto_analysis: 自动分析结果（method="auto" 时需要）

        Returns:
            DataFrame: 缺失值处理后的数据副本
        """
        df = df.copy()

        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue

            # 自动模式下根据分析结果决定方法
            actual_method = method
            if method == "auto" and auto_analysis and col in auto_analysis:
                suggested = auto_analysis[col]["suggested_missing_method"]
                method_map = {"均值": "mean", "中位数": "median", "前向填充": "forward_fill",
                            "后向填充": "backward_fill", "插值": "interpolate", "KNN填充": "knn"}
                actual_method = method_map.get(suggested, "mean")

            if actual_method == "mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
            elif actual_method == "median":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
            elif actual_method == "forward_fill":
                # 先向前填充，若开头仍有缺失则向后补
                df[col].fillna(method='ffill', inplace=True)
                df[col].fillna(method='bfill', inplace=True)
            elif actual_method == "backward_fill":
                df[col].fillna(method='bfill', inplace=True)
                df[col].fillna(method='ffill', inplace=True)
            elif actual_method == "interpolate":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].interpolate(method='linear')
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(method='bfill', inplace=True)
            elif actual_method == "knn":
                # KNN 只对数值列有效，需至少 2 列参与
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1 and col in numeric_cols:
                    numeric_data = df[numeric_cols].copy()
                    imputer = KNNImputer(n_neighbors=min(5, len(numeric_data) - 1))
                    imputed = imputer.fit_transform(numeric_data)
                    df[numeric_cols] = pd.DataFrame(imputed, columns=numeric_cols, index=df.index)
                else:
                    df[col].fillna(df[col].median() if pd.api.types.is_numeric_dtype(df[col])
                                  else df[col].mode()[0] if not df[col].mode().empty else "Unknown",
                                  inplace=True)

        return df

    def detect_outliers(self, df: pd.DataFrame, method: str,
                        threshold: float = 1.5) -> Dict[str, List[int]]:
        """检测异常值

        Args:
            df: 数据 DataFrame
            method: 检测方法 — "iqr"（四分位距法）或 "zscore"（Z-Score 法）
            threshold: IQR 倍数（默认 1.5）或 Z-Score 阈值（默认 3.0）

        Returns:
            dict: 列名 → 异常值行索引列表
        """
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            valid = df[col].dropna()
            if len(valid) < 3:
                continue

            if method == "iqr":
                q1 = valid.quantile(0.25)
                q3 = valid.quantile(0.75)
                iqr = q3 - q1
                if iqr <= 0:
                    continue
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                outlier_mask = (df[col] < lower) | (df[col] > upper)
                outlier_indices = df.index[outlier_mask & df[col].notna()].tolist()
            elif method in ("zscore", "z-score"):
                mean_val = valid.mean()
                std_val = valid.std()
                if std_val <= 0:
                    continue
                z_scores = np.abs((df[col] - mean_val) / std_val)
                outlier_indices = df.index[(z_scores > threshold) & df[col].notna()].tolist()
            else:
                continue

            if outlier_indices:
                outliers[col] = outlier_indices

        return outliers

    def handle_outliers(self, df: pd.DataFrame, handle_method: str,
                        outliers: Dict[str, List[int]]) -> pd.DataFrame:
        """处理异常值

        Args:
            df: 数据 DataFrame
            handle_method: "删除" — 直接删除异常值所在行；"裁剪" — 将异常值限制在 IQR 上下界内
            outliers: detect_outliers() 返回的异常值字典

        Returns:
            DataFrame: 处理后的数据副本
        """
        df = df.copy()

        for col, indices in outliers.items():
            if not indices:
                continue
            if handle_method == "删除":
                df.drop(index=indices, inplace=True)
            elif handle_method == "裁剪":
                # 计算裁剪边界
                valid = df[col].dropna()
                q1 = valid.quantile(0.25)
                q3 = valid.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr if iqr > 0 else valid.min()
                upper = q3 + 1.5 * iqr if iqr > 0 else valid.max()
                df.loc[df.index.isin(indices), col] = df.loc[
                    df.index.isin(indices), col
                ].clip(lower, upper)

        return df

    def normalize(self, df: pd.DataFrame, method: str,
                  range_tuple: Optional[Tuple[float, float]] = None
                  ) -> Tuple[pd.DataFrame, Dict]:
        """数据归一化/标准化

        Args:
            df: 数据 DataFrame
            method: "minmax" — Min-Max 归一化到指定范围；"standard" — Z-Score 标准化
            range_tuple: Min-Max 归一化的目标范围 (min, max)，默认 (0, 1)

        Returns:
            (DataFrame, dict): 归一化后数据和参数字典（用于后续反归一化）
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        params = {}

        if method == "minmax":
            if range_tuple is None:
                range_tuple = (0, 1)
            min_val, max_val = range_tuple
            for col in numeric_cols:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max > col_min:
                    df[col] = (df[col] - col_min) / (col_max - col_min) * (max_val - min_val) + min_val
                else:
                    df[col] = float(min_val)
                params[col] = {"min": col_min, "max": col_max, "range": (min_val, max_val)}

        elif method == "standard":
            for col in numeric_cols:
                col_mean = df[col].mean()
                col_std = df[col].std()
                if col_std > 0:
                    df[col] = (df[col] - col_mean) / col_std
                else:
                    df[col] = 0
                params[col] = {"mean": col_mean, "std": col_std}

        self.norm_params = params
        return df, params
