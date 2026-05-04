from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QGroupBox, QTextEdit, QMessageBox,
                             QTabWidget, QFileDialog, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class PredictionPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.y_true = None
        self.y_pred = None
        self.setMinimumHeight(300)

    def update_data(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

        if self.y_true is None or self.y_pred is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(w//2 - 50, h//2, "等待数据...")
            return

        n = min(500, len(self.y_true))
        indices = np.linspace(0, len(self.y_true) - 1, n).astype(int)

        true_vals = self.y_true[indices].flatten()
        pred_vals = self.y_pred[indices].flatten()

        max_val = max(true_vals.max(), pred_vals.max())
        min_val = min(true_vals.min(), pred_vals.min())

        padding = 50
        draw_w = w - padding * 2
        draw_h = h - padding * 2

        painter.setPen(QPen(QColor(0, 200, 100), 2))
        for i in range(n - 1):
            x1 = padding + i * draw_w / (n - 1)
            x2 = padding + (i + 1) * draw_w / (n - 1)
            y1 = padding + draw_h * (1 - (true_vals[i] - min_val) / (max_val - min_val + 1e-8))
            y2 = padding + draw_h * (1 - (true_vals[i+1] - min_val) / (max_val - min_val + 1e-8))
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        painter.setPen(QPen(QColor(200, 100, 100), 2))
        for i in range(n - 1):
            x1 = padding + i * draw_w / (n - 1)
            x2 = padding + (i + 1) * draw_w / (n - 1)
            y1 = padding + draw_h * (1 - (pred_vals[i] - min_val) / (max_val - min_val + 1e-8))
            y2 = padding + draw_h * (1 - (pred_vals[i+1] - min_val) / (max_val - min_val + 1e-8))
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        painter.setPen(QColor(200, 200, 200))
        painter.drawText(5, padding - 5, f"{max_val:.2f}")
        painter.drawText(5, h - padding + 15, f"{min_val:.2f}")

        legend_y = 20
        painter.setPen(QColor(0, 200, 100))
        painter.drawLine(padding, legend_y, padding + 30, legend_y)
        painter.drawText(padding + 35, legend_y + 5, "真实值")
        painter.setPen(QColor(200, 100, 100))
        painter.drawLine(padding + 120, legend_y, padding + 150, legend_y)
        painter.drawText(padding + 155, legend_y + 5, "预测值")


class ScatterPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.y_true = None
        self.y_pred = None
        self.setMinimumHeight(300)

    def update_data(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

        if self.y_true is None or self.y_pred is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(w//2 - 50, h//2, "等待数据...")
            return

        true_vals = self.y_true.flatten()
        pred_vals = self.y_pred.flatten()

        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())

        padding = 50
        draw_w = w - padding * 2
        draw_h = h - padding * 2

        painter.setPen(QColor(100, 200, 100))
        for i in range(len(true_vals)):
            x = padding + (true_vals[i] - min_val) / (max_val - min_val + 1e-8) * draw_w
            y = padding + (1 - (pred_vals[i] - min_val) / (max_val - min_val + 1e-8)) * draw_h
            painter.drawEllipse(int(x), int(y), 3, 3)

        painter.setPen(QColor(255, 255, 255))
        painter.drawLine(padding, h - padding, w - padding, padding)

        painter.setPen(QColor(200, 200, 200))
        painter.drawText(5, padding - 5, f"{max_val:.2f}")
        painter.drawText(5, h - padding + 15, f"{min_val:.2f}")


class ResultPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.results = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)

        self.save_metrics_btn = QPushButton("保存评估指标")
        self.save_metrics_btn.clicked.connect(self.save_metrics)
        metrics_layout.addWidget(self.save_metrics_btn)

        tabs.addTab(metrics_tab, "评估指标")

        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)

        self.pred_plot = PredictionPlot()
        plot_layout.addWidget(QLabel("预测对比曲线:"))
        plot_layout.addWidget(self.pred_plot)

        self.scatter_plot = ScatterPlot()
        plot_layout.addWidget(QLabel("真实值 vs 预测值 散点图:"))
        plot_layout.addWidget(self.scatter_plot)

        tabs.addTab(plot_tab, "可视化")

        layout.addWidget(tabs)

        bottom_layout = QHBoxLayout()
        self.btn_export = QPushButton("导出模型")
        self.btn_export.clicked.connect(self.export_model)
        bottom_layout.addWidget(self.btn_export)

        self.btn_export_full = QPushButton("导出完整项目")
        self.btn_export_full.clicked.connect(self.export_full)
        bottom_layout.addWidget(self.btn_export_full)

        bottom_layout.addWidget(QWidget(), 1)
        layout.addLayout(bottom_layout)

    def set_results(self, results):
        self.results = results

        config = results["model_config"]
        info = f"模型配置\n"
        info += f"{'='*50}\n"
        info += f"模型类型: {config['model_type'].upper()}\n"
        info += f"任务类型: {config['task_type']}\n"
        info += f"输入特征: {config['input_size']}\n"
        info += f"输出目标: {config['output_size']}\n"
        info += f"优化器: {config['optimizer']}\n"
        info += f"学习率: {config['learning_rate']}\n"
        info += f"批次大小: {config['batch_size']}\n"
        info += f"训练轮次: {config['epochs']}\n\n"

        metrics = results["metrics"]
        info += f"评估指标\n"
        info += f"{'='*50}\n"
        if config["task_type"] in ("回归", "regression"):
            for key, value in metrics.items():
                info += f"{key.upper()}: {value:.6f}\n"
        else:
            for key, value in metrics.items():
                info += f"{key.upper()}: {value:.4f}\n"

        self.metrics_text.setPlainText(info)

        self.pred_plot.update_data(results["y_true"], results["y_pred"])
        self.scatter_plot.update_data(results["y_true"], results["y_pred"])

    def save_metrics(self):
        if not self.results:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存评估指标", "", "Text Files (*.txt)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.metrics_text.toPlainText())
            QMessageBox.information(self, "成功", "评估指标已保存")

    def export_model(self):
        if not self.results:
            QMessageBox.warning(self, "警告", "没有可导出的模型")
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存模型", "", "PyTorch Models (*.pt *.pth)")
        if path:
            torch.save({
                'model_state_dict': self.results["model"].state_dict(),
                'model_config': self.results["model_config"],
                'metrics': self.results["metrics"]
            }, path)
            QMessageBox.information(self, "成功", "模型已导出")

    def export_full(self):
        if not self.results:
            QMessageBox.warning(self, "警告", "没有可导出的内容")
            return

        folder = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if folder:
            import os
            model_path = os.path.join(folder, "model.pth")
            torch.save({
                'model_state_dict': self.results["model"].state_dict(),
                'model_config': self.results["model_config"],
                'metrics': self.results["metrics"]
            }, model_path)

            metrics_path = os.path.join(folder, "metrics.txt")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write(self.metrics_text.toPlainText())

            config_path = os.path.join(folder, "config.json")
            import json
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.results["model_config"], f, indent=2, ensure_ascii=False)

            QMessageBox.information(self, "成功", f"完整项目已导出到: {folder}")