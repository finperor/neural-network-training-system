# ┌────────────────────────────────────────────────────────────┐
# │  模型训练器                                                │
# │  封装训练循环、验证、早停、学习率调度、检查点保存等功能      │
# └────────────────────────────────────────────────────────────┘

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
import numpy as np
import time
from pathlib import Path
import json


class Trainer:
    """神经网络模型训练器

    功能：
    - 单轮训练 (train_one_epoch) 与验证 (validate)
    - 完整训练循环 (train) 含早停与学习率调度
    - 模型检查点的保存与加载
    - 自动追踪训练/验证损失历史

    典型用法（逐步控制）：
        trainer.train_one_epoch()
        trainer.validate()
        # 外部更新 best_val_loss 和 best_model_state

    典型用法（自动训练）：
        trainer.train(epochs=100)
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device,
                 optimizer: str = "adam", lr: float = 0.001,
                 loss_fn: str = "mse", patience: int = 10,
                 gradient_clip: Optional[float] = None):
        """
        Args:
            model: PyTorch 模型实例
            train_loader: 训练数据 DataLoader
            val_loader: 验证数据 DataLoader
            device: 计算设备 (cpu / cuda)
            optimizer: 优化器名称 ("adam"/"sgd"/"rmsprop"/"adamw")
            lr: 初始学习率
            loss_fn: 损失函数名称 ("mse"/"mae"/"huber"/"cross_entropy")
            patience: 早停耐心值（轮数）
            gradient_clip: 梯度裁剪阈值，None 表示不裁剪
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.gradient_clip = gradient_clip

        # 最佳验证损失，初始为无穷大
        self.best_val_loss = float('inf')
        self.best_model_state = None

        # 损失历史记录
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.current_epoch = 0

        # 构建优化器和损失函数
        self.optimizer = self._get_optimizer(optimizer, lr)
        self.criterion = self._get_loss_function(loss_fn)

        self.scheduler = None   # 学习率调度器

    # ── 内部辅助方法 ─────────────────────────────────────────

    def _get_optimizer(self, name: str, lr: float) -> torch.optim.Optimizer:
        """根据名称构建优化器"""
        optimizers = {
            "adam": torch.optim.Adam(self.model.parameters(), lr=lr),
            "sgd": torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9),
            "rmsprop": torch.optim.RMSprop(self.model.parameters(), lr=lr),
            "adamw": torch.optim.AdamW(self.model.parameters(), lr=lr)
        }
        return optimizers.get(name.lower(), torch.optim.Adam(self.model.parameters(), lr=lr))

    def _get_loss_function(self, name: str) -> nn.Module:
        """根据名称构建损失函数"""
        losses = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
            "huber": nn.HuberLoss(),
            "cross_entropy": nn.CrossEntropyLoss()
        }
        return losses.get(name.lower(), nn.MSELoss())

    def set_scheduler(self, scheduler_type: str, **kwargs):
        """设置学习率调度器

        Args:
            scheduler_type: "step"/"exponential"/"cosine"/"plateau"
            **kwargs: 对应的调度器参数
        """
        if scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=kwargs.get("step_size", 10),
                gamma=kwargs.get("gamma", 0.1)
            )
        elif scheduler_type == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=kwargs.get("gamma", 0.95)
            )
        elif scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=kwargs.get("T_max", 100)
            )
        elif scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 5)
            )

    # ── 训练核心循环 ─────────────────────────────────────────

    def train_one_epoch(self) -> Dict[str, float]:
        """执行单个训练轮次

        遍历训练集所有批次，计算损失、反向传播并更新参数。

        Returns:
            dict: {"train_loss": 本批平均损失值}
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()                  # 清零梯度
            outputs = self.model(batch_x)               # 前向传播
            loss = self.criterion(outputs, batch_y)     # 计算损失
            loss.backward()                             # 反向传播

            if self.gradient_clip:
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()                       # 更新参数
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {"train_loss": avg_loss}

    def validate(self) -> Dict[str, float]:
        """执行验证

        在验证集上评估模型，不计算梯度。

        Returns:
            dict: {"val_loss": 本批平均验证损失}
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {"val_loss": avg_loss}

    # ── 完整训练循环 ─────────────────────────────────────────

    def train(self, epochs: int, callback: Optional[Callable] = None,
              checkpoint_path: Optional[str] = None) -> Dict:
        """执行完整训练循环（含早停与学习率调度）

        该方法是自包含的：自动管理 best_val_loss、早停计数、
        学习率调度以及定期检查点保存。

        Args:
            epochs: 最大训练轮数
            callback: 每轮结束后的回调函数 callback(metrics_dict)
            checkpoint_path: 检查点保存路径（每 10 轮保存一次）

        Returns:
            dict: 训练结果摘要
        """
        start_time = time.time()
        no_improve_count = 0

        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()

            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()

            # 记录损失历史
            self.train_losses.append(train_metrics["train_loss"])
            self.val_losses.append(val_metrics["val_loss"])

            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # 早停逻辑：验证损失未改善则计数
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_model_state = self.model.state_dict().copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            metrics = {
                "epoch": self.current_epoch,
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_metrics["val_loss"],
                "learning_rate": current_lr,
                "epoch_time": epoch_time
            }

            if callback:
                callback(metrics)

            # 定期保存检查点
            if checkpoint_path and (epoch + 1) % 10 == 0:
                self.save_checkpoint(checkpoint_path, epoch + 1)

            if no_improve_count >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        total_time = time.time() - start_time

        # 恢复最佳模型状态
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "total_time": total_time,
            "total_epochs": self.current_epoch
        }

    # ── 检查点管理 ───────────────────────────────────────────

    def save_checkpoint(self, path: str, epoch: int):
        """保存训练检查点

        Args:
            path: 保存路径
            epoch: 当前轮数
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载训练检查点并恢复状态

        Args:
            path: 检查点文件路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.current_epoch = checkpoint["epoch"]
