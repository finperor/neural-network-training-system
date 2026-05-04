import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QStatusBar, QMenuBar)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QAction
from ui.data_panel import DataPanel
from ui.preprocess_panel import PreprocessPanel
from ui.model_panel import ModelPanel
from ui.train_panel import TrainPanel
from ui.result_panel import ResultPanel
from core.device_manager import DeviceManager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("化工过程神经网络训练系统")
        self.setGeometry(100, 100, 1400, 800)
        self.devices = DeviceManager.get_available_devices()
        self.setup_ui()
        self.setup_menu()
        self.setup_statusbar()

    def setup_ui(self):
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.West)

        self.data_panel = DataPanel()
        self.preprocess_panel = PreprocessPanel()
        self.model_panel = ModelPanel()
        self.train_panel = TrainPanel(self.devices)
        self.result_panel = ResultPanel()

        self.tabs.addTab(self.data_panel, "数据导入")
        self.tabs.addTab(self.preprocess_panel, "数据预处理")
        self.tabs.addTab(self.model_panel, "模型配置")
        self.tabs.addTab(self.train_panel, "模型训练")
        self.tabs.addTab(self.result_panel, "结果分析")

        self.data_panel.data_ready.connect(self.on_data_ready)
        self.data_panel.data_loaded.connect(self.on_data_loaded)
        self.preprocess_panel.preprocess_complete.connect(self.on_preprocess_complete)
        self.model_panel.model_ready.connect(self.on_model_ready)
        self.train_panel.training_complete.connect(self.on_training_complete)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central_widget)

    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("帮助")
        about_action = QAction("关于", self)
        help_menu.addAction(about_action)

    def setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("就绪")
        self.update_device_status()

    def update_device_status(self):
        if self.devices["cuda"]:
            device_info = self.devices["cuda_info"]
            status = f"GPU: {device_info['device_name']} | 显存: {device_info['memory_allocated']:.1f}GB"
        else:
            status = "CPU模式"
        self.statusbar.showMessage(status)

    def on_data_ready(self, data_info):
        self.model_panel.set_data(data_info)

    def on_data_loaded(self, df, data_info):
        self.preprocess_panel.set_raw_data(df, data_info)

    def on_preprocess_complete(self, preprocessed_data):
        self.model_panel.set_preprocessed_data(preprocessed_data)

    def on_model_ready(self, model_config):
        self.train_panel.set_model_config(model_config)

    def on_training_complete(self, results):
        self.result_panel.set_results(results)
        self.tabs.setCurrentIndex(4)