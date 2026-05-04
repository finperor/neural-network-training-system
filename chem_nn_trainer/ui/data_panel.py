from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QTableWidget, QTableWidgetItem, QFileDialog,
                             QGroupBox, QTextEdit, QMessageBox, QSplitter)
from PyQt6.QtCore import pyqtSignal, Qt
from data.data_loader import DataLoader
import pandas as pd
import os


class DataPanel(QWidget):
    data_ready = pyqtSignal(dict)
    data_loaded = pyqtSignal(object, dict)

    def __init__(self):
        super().__init__()
        self.data_loader = DataLoader()
        self.current_df = None
        self.data_info = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        file_group = QGroupBox("数据文件")
        file_layout = QHBoxLayout()

        self.btn_load = QPushButton("加载CSV文件")
        self.btn_load.clicked.connect(self.load_file)
        self.btn_preview = QPushButton("数据预览")
        self.btn_preview.clicked.connect(self.preview_data)
        self.btn_stats = QPushButton("数据统计")
        self.btn_stats.clicked.connect(self.show_stats)

        self.lbl_file = QLabel("未加载文件")
        self.lbl_file.setStyleSheet("color: #666;")

        file_layout.addWidget(self.btn_load)
        file_layout.addWidget(self.btn_preview)
        file_layout.addWidget(self.btn_stats)
        file_layout.addWidget(self.lbl_file, 1)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.table_preview = QTableWidget()
        self.table_preview.setMaximumHeight(200)
        splitter.addWidget(self.table_preview)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        splitter.addWidget(self.stats_text)

        layout.addWidget(splitter)

        self.col_select_group = QGroupBox("列选择")
        col_layout = QVBoxLayout()

        self.col_info_text = QTextEdit()
        self.col_info_text.setReadOnly(True)
        self.col_info_text.setMaximumHeight(150)
        col_layout.addWidget(self.col_info_text)
        self.col_select_group.setLayout(col_layout)
        layout.addWidget(self.col_select_group)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择CSV文件", "", "CSV Files (*.csv)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if file_path:
            try:
                self.current_df, self.data_info = self.data_loader.load_csv(file_path)
                self.lbl_file.setText(os.path.basename(file_path))
                self.data_loaded.emit(self.current_df, self.data_info)
                self.data_ready.emit(self.data_info)
                QMessageBox.information(self, "成功", f"成功加载数据: {self.data_info['rows']} 行, {self.data_info['columns']} 列")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败: {str(e)}")

    def preview_data(self):
        if self.current_df is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return
        self.table_preview.setRowCount(min(20, len(self.current_df)))
        self.table_preview.setColumnCount(len(self.current_df.columns))
        self.table_preview.setHorizontalHeaderLabels(list(self.current_df.columns))

        for i, row in self.current_df.head(20).iterrows():
            for j, val in enumerate(row):
                self.table_preview.setItem(i, j, QTableWidgetItem(str(val)))

    def show_stats(self):
        if self.current_df is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return

        stats = self.data_loader.get_all_stats()
        info_text = "列名 | 缺失值 | 缺失率% | 均值 | 标准差 | 最小值 | 最大值\n"
        info_text += "=" * 80 + "\n"

        for col, stat in stats.items():
            if stat.get('mean') is not None:
                info_text += f"{col[:15]:15} | {stat['missing']:4} | {stat['missing_rate']:5.1f} | "
                info_text += f"{stat['mean']:8.2f} | {stat['std']:8.2f} | {stat['min']:8.2f} | {stat['max']:8.2f}\n"

        self.col_info_text.setPlainText(info_text)