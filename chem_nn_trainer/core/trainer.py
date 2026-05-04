import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
import numpy as np
import time
from pathlib import Path
import json


class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device,
                 optimizer: str = "adam", lr: float = 0.001,
                 loss_fn: str = "mse", patience: int = 10,
                 gradient_clip: Optional[float] = None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.gradient_clip = gradient_clip
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.current_epoch = 0

        self.optimizer = self._get_optimizer(optimizer, lr)
        self.criterion = self._get_loss_function(loss_fn)
        self.scheduler = None

    def _get_optimizer(self, name: str, lr: float) -> torch.optim.Optimizer:
        optimizers = {
            "adam": torch.optim.Adam(self.model.parameters(), lr=lr),
            "sgd": torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9),
            "rmsprop": torch.optim.RMSprop(self.model.parameters(), lr=lr),
            "adamw": torch.optim.AdamW(self.model.parameters(), lr=lr)
        }
        return optimizers.get(name.lower(), torch.optim.Adam(self.model.parameters(), lr=lr))

    def _get_loss_function(self, name: str) -> nn.Module:
        losses = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
            "huber": nn.HuberLoss(),
            "cross_entropy": nn.CrossEntropyLoss()
        }
        return losses.get(name.lower(), nn.MSELoss())

    def set_scheduler(self, scheduler_type: str, **kwargs):
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

    def train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()

            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}

    def validate(self) -> Dict[str, float]:
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

        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}

    def train(self, epochs: int, callback: Optional[Callable] = None,
              checkpoint_path: Optional[str] = None) -> Dict:
        start_time = time.time()
        no_improve_count = 0

        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()

            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()

            self.train_losses.append(train_metrics["train_loss"])
            self.val_losses.append(val_metrics["val_loss"])

            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

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

            if checkpoint_path and (epoch + 1) % 10 == 0:
                self.save_checkpoint(checkpoint_path, epoch + 1)

            if no_improve_count >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        total_time = time.time() - start_time

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "total_time": total_time,
            "total_epochs": self.current_epoch
        }

    def save_checkpoint(self, path: str, epoch: int):
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
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.current_epoch = checkpoint["epoch"]