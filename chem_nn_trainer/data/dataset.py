# ┌────────────────────────────────────────────────────────────┐
# │  数据集定义与数据加载                                       │
# │  定义 PyTorch Dataset 子类及数据集划分逻辑                  │
# └────────────────────────────────────────────────────────────┘

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class SimpleDataset(Dataset):
    """简单数据集（用于 MLP 等非时序模型）

    直接包装特征张量和目标张量，按索引返回 (features, targets) 对。
    """

    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features      # 形状: (N, D_in)
        self.targets = targets        # 形状: (N, D_out)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class TimeSeriesDataset(Dataset):
    """时序数据集（用于 RNN/LSTM/GRU/CNN-1D 等时序模型）

    将原始数据按滑动窗口切分为 (序列, 下一时刻目标) 的形式。
    """

    def __init__(self, data: np.ndarray, target_indices: List[int], seq_length: int):
        """
        Args:
            data: 原始数据数组，形状 (N, C)
            target_indices: 目标列在数据中的列索引列表
            seq_length: 滑动窗口长度
        """
        self.seq_length = seq_length
        self.target_indices = target_indices
        # 特征列 = 所有列 - 目标列，保证输入不包含未来目标值
        self.feature_indices = [i for i in range(data.shape[1]) if i not in target_indices]

        self.X = []
        self.y = []

        for i in range(len(data) - seq_length):
            # 取 seq_length 长度的特征列作为输入
            seq = data[i:i + seq_length, self.feature_indices]
            # 取下一时刻的目标值作为预测目标
            next_target = data[i + seq_length, target_indices]
            self.X.append(seq)
            self.y.append(next_target)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class DatasetSplitter:
    """数据集划分器

    负责将 DataFrame 按比例划分为训练集、验证集、测试集，
    并创建对应的 PyTorch DataLoader。
    """

    def __init__(self):
        self.X_data = None               # 预留：原始特征数组
        self.y_data = None               # 预留：原始目标数组
        self.split_indices = {}          # 划分后的行索引字典 {train: [...], val: [...], test: [...]}
        self.df = None                   # 原始数据
        self.target_cols = None          # 目标列名列表

    def split(self, df: pd.DataFrame, target_cols: List[str],
              train_ratio: float, val_ratio: float, test_ratio: float,
              is_timeseries: bool = False, seq_length: int = 10) -> Dict:
        """按比例随机划分数据集索引

        Args:
            df: 完整数据 DataFrame
            target_cols: 目标变量列名列表
            train_ratio/val_ratio/test_ratio: 训练/验证/测试集比例
            is_timeseries: 是否为时序数据（保留供扩展使用）
            seq_length: 序列长度（保留供扩展使用）

        Returns:
            dict: 各数据集的索引数组
        """
        self.df = df
        self.target_cols = target_cols

        n = len(df)
        # 随机打乱索引，保证各集合分布均匀
        indices = np.random.permutation(n)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        self.split_indices = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:]
        }

        return self.split_indices

    def create_dataloaders(self, batch_size: int,
                           is_timeseries: bool = False,
                           target_indices: Optional[List[int]] = None,
                           seq_length: int = 10) -> Dict[str, DataLoader]:
        """根据划分结果创建 PyTorch DataLoader

        Args:
            batch_size: 批次大小
            is_timeseries: 是否使用时序数据集格式
            target_indices: 目标列的列索引（未提供则从 target_cols 推断）
            seq_length: 时序序列长度

        Returns:
            dict: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
        """
        if self.df is None:
            raise ValueError("必须先调用 split() 再进行 create_dataloaders()")

        if target_indices is None:
            target_indices = [list(self.df.columns).index(col) for col in self.target_cols]

        # 统一转为 NumPy float32 数组
        data_array = self.df.values.astype(np.float32)

        dataloaders = {}
        for split_name, indices in self.split_indices.items():
            if len(indices) == 0:
                dataloaders[split_name] = None
                continue

            split_data = data_array[indices]

            if is_timeseries and seq_length > 1 and len(split_data) > seq_length:
                dataset = TimeSeriesDataset(split_data, target_indices, seq_length)
            else:
                # 非时序模型：特征列 = 全部列去掉目标列
                feature_indices = [i for i in range(split_data.shape[1]) if i not in target_indices]
                X = split_data[:, feature_indices]
                y = split_data[:, target_indices]
                dataset = SimpleDataset(
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32)
                )

            # 训练集打乱顺序，验证/测试集保持原序
            dataloaders[split_name] = DataLoader(
                dataset, batch_size=batch_size, shuffle=(split_name == "train")
            )

        # 如果验证集为空，回退使用测试集
        if dataloaders.get("val") is None:
            dataloaders["val"] = dataloaders["test"]

        return dataloaders
