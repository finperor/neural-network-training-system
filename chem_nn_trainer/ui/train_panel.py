from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
                             QCheckBox, QTextEdit, QProgressBar, QMessageBox, QTabWidget,
                             QTableWidget, QTableWidgetItem, QFileDialog)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen
import torch
import numpy as np
import time
from data.dataset import DatasetSplitter
from models.nn_builder import ModelBuilder
from models.evaluator import ModelEvaluator
from core.trainer import Trainer
from core.device_manager import DeviceManager


class TrainingMonitor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.train_losses = []
        self.val_losses = []
        self.setMinimumHeight(250)

    def update_data(self, train_losses, val_losses):
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

        if not self.train_losses and not self.val_losses:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(w//2 - 50, h//2, "等待训练...")
            return

        max_loss = max(max(self.train_losses, default=1), max(self.val_losses, default=1))
        min_loss = min(min(self.train_losses, default=0), min(self.val_losses, default=0))
        if max_loss == min_loss:
            max_loss += 1

        padding = 40
        draw_w = w - padding * 2
        draw_h = h - padding * 2

        painter.setPen(QColor(100, 100, 100))
        for i in range(5):
            y = int(padding + draw_h * i / 4)
            painter.drawLine(padding, y, w - padding, y)

        painter.setPen(QColor(200, 200, 200))
        painter.drawText(5, padding - 5, f"{max_loss:.4f}")
        painter.drawText(5, h - padding + 15, f"{min_loss:.4f}")

        if len(self.train_losses) > 1:
            painter.setPen(QPen(QColor(0, 200, 100), 2))
            points = []
            for i, loss in enumerate(self.train_losses):
                x = padding + i * draw_w / (len(self.train_losses) - 1)
                y = padding + draw_h * (1 - (loss - min_loss) / (max_loss - min_loss))
                points.append((x, y))
            for i in range(len(points) - 1):
                painter.drawLine(int(points[i][0]), int(points[i][1]),
                                int(points[i+1][0]), int(points[i+1][1]))

        if len(self.val_losses) > 1:
            painter.setPen(QPen(QColor(200, 100, 100), 2))
            points = []
            for i, loss in enumerate(self.val_losses):
                x = padding + i * draw_w / (len(self.val_losses) - 1)
                y = padding + draw_h * (1 - (loss - min_loss) / (max_loss - min_loss))
                points.append((x, y))
            for i in range(len(points) - 1):
                painter.drawLine(int(points[i][0]), int(points[i][1]),
                                int(points[i+1][0]), int(points[i+1][1]))

        legend_y = 20
        painter.setPen(QColor(0, 200, 100))
        painter.drawLine(padding, legend_y, padding + 30, legend_y)
        painter.drawText(padding + 35, legend_y + 5, "训练损失")
        painter.setPen(QColor(200, 100, 100))
        painter.drawLine(padding + 120, legend_y, padding + 150, legend_y)
        painter.drawText(padding + 155, legend_y + 5, "验证损失")


class TrainPanel(QWidget):
    training_complete = pyqtSignal(dict)

    def __init__(self, devices):
        super().__init__()
        self.devices = devices
        self.model_config = None
        self.trainer = None
        self.is_training = False
        self.current_device = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        config_group = QGroupBox("硬件与数据配置")
        config_layout = QHBoxLayout()

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("计算设备:"))
        self.cb_device = QComboBox()
        device_info = []
        if self.devices["cuda"]:
            device_info.append(f"GPU: {self.devices['cuda_info']['device_name']}")
        device_info.append("CPU")
        self.cb_device.addItems(device_info)
        device_row.addWidget(self.cb_device)

        if self.devices["cuda"]:
            mem_info = self.devices["cuda_info"]
            device_row.addWidget(QLabel(f"显存: {mem_info['memory_allocated']:.1f}GB"))
        config_layout.addLayout(device_row)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("训练集:"))
        self.sb_train = QSpinBox()
        self.sb_train.setRange(50, 90)
        self.sb_train.setValue(70)
        split_row.addWidget(self.sb_train)
        split_row.addWidget(QLabel("%"))
        split_row.addWidget(QLabel("验证集:"))
        self.sb_val = QSpinBox()
        self.sb_val.setRange(5, 30)
        self.sb_val.setValue(15)
        split_row.addWidget(self.sb_val)
        split_row.addWidget(QLabel("%"))
        split_row.addWidget(QLabel("测试集:"))
        self.sb_test = QSpinBox()
        self.sb_test.setRange(5, 30)
        self.sb_test.setValue(15)
        split_row.addWidget(self.sb_test)
        split_row.addWidget(QLabel("%"))
        config_layout.addLayout(split_row)

        seq_row = QHBoxLayout()
        seq_row.addWidget(QLabel("序列长度(时序模型):"))
        self.sb_seq_length = QSpinBox()
        self.sb_seq_length.setRange(1, 100)
        self.sb_seq_length.setValue(10)
        seq_row.addWidget(self.sb_seq_length)
        self.cb_timeseries = QCheckBox("时序划分")
        self.cb_timeseries.setChecked(True)
        seq_row.addWidget(self.cb_timeseries)
        config_layout.addLayout(seq_row)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        self.monitor = TrainingMonitor()
        layout.addWidget(self.monitor)

        metrics_group = QGroupBox("训练指标")
        metrics_layout = QVBoxLayout()
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(100)
        metrics_layout.addWidget(self.metrics_text)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        progress_layout = QHBoxLayout()
        self.progress = QProgressBar()
        progress_layout.addWidget(self.progress)
        self.lbl_progress = QLabel("0/0")
        progress_layout.addWidget(self.lbl_progress)
        layout.addLayout(progress_layout)

        button_layout = QHBoxLayout()
        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self.start_training)
        button_layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("暂停")
        self.btn_pause.clicked.connect(self.pause_training)
        self.btn_pause.setEnabled(False)
        button_layout.addWidget(self.btn_pause)

        self.btn_resume = QPushButton("继续")
        self.btn_resume.clicked.connect(self.resume_training)
        self.btn_resume.setEnabled(False)
        button_layout.addWidget(self.btn_resume)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        button_layout.addWidget(self.btn_stop)

        button_layout.addWidget(QWidget(), 1)
        layout.addLayout(button_layout)

    def set_model_config(self, config):
        self.model_config = config
        info = f"模型类型: {config['model_type'].upper()}\n"
        info += f"输入: {config['input_size']}, 输出: {config['output_size']}\n"
        info += f"优化器: {config['optimizer']}, 学习率: {config['learning_rate']}\n"
        info += f"批次: {config['batch_size']}, 轮次: {config['epochs']}"
        QMessageBox.information(self, "模型配置", info)

    def start_training(self):
        if not self.model_config:
            QMessageBox.warning(self, "警告", "请先完成模型配置")
            return

        device_name = self.cb_device.currentText()
        if "GPU" in device_name and self.devices["cuda"]:
            self.current_device = torch.device("cuda")
        else:
            self.current_device = torch.device("cpu")

        df = self.model_config["data"]["df"]
        target_cols = self.model_config["target_columns"]
        train_ratio = self.sb_train.value() / 100
        val_ratio = self.sb_val.value() / 100

        splitter = DatasetSplitter()
        split_data = splitter.split(df, target_cols, train_ratio, val_ratio, val_ratio,
                                    is_timeseries=self.cb_timeseries.isChecked(),
                                    seq_length=self.sb_seq_length.value())

        is_timeseries = self.model_config["model_type"] in ["rnn", "lstm", "gru", "cnn1d"]
        target_indices = [list(df.columns).index(col) for col in target_cols]

        dataloaders = splitter.create_dataloaders(
            self.model_config["batch_size"],
            is_timeseries=is_timeseries,
            target_indices=target_indices,
            seq_length=self.sb_seq_length.value()
        )

        self.model = ModelBuilder.build(
            self.model_config["model_type"],
            self.model_config["input_size"],
            self.model_config["output_size"],
            {k: v for k, v in self.model_config.items()
             if k not in ["task_type", "model_type", "input_size", "output_size",
                         "target_columns", "optimizer", "learning_rate", "batch_size",
                         "epochs", "loss_fn", "scheduler", "dropout", "patience",
                         "data", "reg_lambda", "regularization"]}
        )

        self.trainer = Trainer(
            self.model,
            dataloaders["train"],
            dataloaders["val"],
            self.current_device,
            optimizer=self.model_config["optimizer"],
            lr=self.model_config["learning_rate"],
            loss_fn=self.model_config["loss_fn"],
            patience=self.model_config["patience"]
        )

        if self.model_config["scheduler"] != "无":
            scheduler_map = {"阶梯衰减": "step", "指数衰减": "exponential",
                          "余弦退火": "cosine", "ReduceLROnPlateau": "plateau"}
            self.trainer.set_scheduler(scheduler_map[self.model_config["scheduler"]])

        self.is_training = True
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

        self.epochs = self.model_config["epochs"]
        self.current_epoch = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_training)
        self.timer.start(500)

        self.train_thread = None

    def update_training(self):
        if self.current_epoch >= self.epochs:
            self.finish_training()
            return

        metrics = self.trainer.train_one_epoch()
        val_metrics = self.trainer.validate()

        self.trainer.train_losses.append(metrics['train_loss'])
        self.trainer.val_losses.append(val_metrics['val_loss'])

        if val_metrics['val_loss'] < self.trainer.best_val_loss:
            self.trainer.best_val_loss = val_metrics['val_loss']
            self.trainer.best_model_state = self.trainer.model.state_dict().copy()

        self.current_epoch += 1
        self.progress.setValue(int(self.current_epoch / self.epochs * 100))
        self.lbl_progress.setText(f"{self.current_epoch}/{self.epochs}")

        self.monitor.update_data(self.trainer.train_losses, self.trainer.val_losses)

        info = f"轮次: {self.current_epoch}/{self.epochs}\n"
        info += f"训练损失: {metrics['train_loss']:.6f}\n"
        info += f"验证损失: {val_metrics['val_loss']:.6f}\n"
        info += f"当前学习率: {self.trainer.optimizer.param_groups[0]['lr']:.6f}"
        self.metrics_text.setPlainText(info)

    def pause_training(self):
        self.is_training = False
        self.timer.stop()
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)

    def resume_training(self):
        self.is_training = True
        self.timer.start(500)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)

    def stop_training(self):
        self.timer.stop()
        self.is_training = False
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)
        QMessageBox.information(self, "停止", "训练已停止")

    def finish_training(self):
        self.timer.stop()

        if self.trainer.best_model_state is not None:
            self.trainer.model.load_state_dict(self.trainer.best_model_state)

        test_loader = self.trainer.val_loader
        y_true, y_pred = ModelEvaluator.predict(self.trainer.model, test_loader, self.current_device)

        task_type = self.model_config["task_type"]
        metrics = ModelEvaluator.evaluate(y_true.flatten(), y_pred.flatten(), task_type.lower())

        results = {
            "model": self.trainer.model,
            "device": self.current_device,
            "train_losses": self.trainer.train_losses,
            "val_losses": self.trainer.val_losses,
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "model_config": self.model_config,
            "total_time": sum(self.trainer.val_losses) * 0.1
        }

        self.training_complete.emit(results)

        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)

        info = "训练完成!\n\n"
        info += f"最终验证损失: {self.trainer.best_val_loss:.6f}\n\n"
        if self.model_config["task_type"] in ("回归", "regression"):
            info += f"MSE: {metrics.get('mse', 0):.6f}\n"
            info += f"RMSE: {metrics.get('rmse', 0):.6f}\n"
            info += f"MAE: {metrics.get('mae', 0):.6f}\n"
            info += f"R²: {metrics.get('r2', 0):.4f}"
        else:
            info += f"准确率: {metrics.get('accuracy', 0):.4f}\n"
            info += f"F1: {metrics.get('f1', 0):.4f}"

        QMessageBox.information(self, "训练完成", info)