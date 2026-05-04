from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
                             QCheckBox, QTextEdit, QProgressBar, QMessageBox, QScrollArea)
from PyQt6.QtCore import pyqtSignal, Qt
from data.preprocessor import DataPreprocessor
import pandas as pd


class PreprocessPanel(QWidget):
    preprocess_complete = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.raw_data = None
        self.data_info = None
        self.preprocessor = DataPreprocessor()
        self.preprocessed_df = None
        self.auto_analysis = None
        self.setup_ui()

    def set_raw_data(self, df, data_info):
        self.raw_data = df
        self.data_info = data_info
        self.current_df = df.copy()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        missing_group = QGroupBox("缺失值处理")
        missing_layout = QVBoxLayout()

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("处理方法:"))
        self.cb_missing_method = QComboBox()
        self.cb_missing_method.addItems(["自动", "均值", "中位数", "前向填充", "后向填充", "插值", "KNN填充"])
        method_row.addWidget(self.cb_missing_method)
        missing_layout.addLayout(method_row)

        self.btn_analyze = QPushButton("自动分析数据质量")
        self.btn_analyze.clicked.connect(self.analyze_data)
        missing_layout.addWidget(self.btn_analyze)

        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMaximumHeight(150)
        missing_layout.addWidget(QLabel("分析结果:"))
        missing_layout.addWidget(self.analysis_text)

        missing_group.setLayout(missing_layout)
        scroll_layout.addWidget(missing_group)

        outlier_group = QGroupBox("异常值处理")
        outlier_layout = QVBoxLayout()

        detect_row = QHBoxLayout()
        detect_row.addWidget(QLabel("检测方法:"))
        self.cb_outlier_method = QComboBox()
        self.cb_outlier_method.addItems(["IQR", "Z-Score"])
        detect_row.addWidget(self.cb_outlier_method)
        detect_row.addWidget(QLabel("阈值:"))
        self.sb_outlier_threshold = QDoubleSpinBox()
        self.sb_outlier_threshold.setRange(1.0, 5.0)
        self.sb_outlier_threshold.setValue(1.5)
        self.sb_outlier_threshold.setSingleStep(0.1)
        detect_row.addWidget(self.sb_outlier_threshold)
        outlier_layout.addLayout(detect_row)

        handle_row = QHBoxLayout()
        handle_row.addWidget(QLabel("处理方式:"))
        self.cb_outlier_handle = QComboBox()
        self.cb_outlier_handle.addItems(["删除", "裁剪"])
        handle_row.addWidget(self.cb_outlier_handle)

        self.btn_detect = QPushButton("检测异常值")
        self.btn_detect.clicked.connect(self.detect_outliers)
        handle_row.addWidget(self.btn_detect)
        outlier_layout.addLayout(handle_row)

        self.outlier_text = QTextEdit()
        self.outlier_text.setReadOnly(True)
        self.outlier_text.setMaximumHeight(100)
        outlier_layout.addWidget(QLabel("检测结果:"))
        outlier_layout.addWidget(self.outlier_text)

        outlier_group.setLayout(outlier_layout)
        scroll_layout.addWidget(outlier_group)

        norm_group = QGroupBox("数据归一化")
        norm_layout = QVBoxLayout()

        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel("归一化方法:"))
        self.cb_norm_method = QComboBox()
        self.cb_norm_method.addItems(["Min-Max归一化", "Z-Score标准化", "不归一化"])
        norm_row.addWidget(self.cb_norm_method)

        self.sb_norm_min = QDoubleSpinBox()
        self.sb_norm_min.setRange(0, 0)
        self.sb_norm_min.setValue(0)
        norm_row.addWidget(QLabel("范围下限:"))
        norm_row.addWidget(self.sb_norm_min)

        self.sb_norm_max = QDoubleSpinBox()
        self.sb_norm_max.setRange(1, 100)
        self.sb_norm_max.setValue(1)
        norm_row.addWidget(QLabel("范围上限:"))
        norm_row.addWidget(self.sb_norm_max)

        norm_layout.addLayout(norm_row)

        norm_group.setLayout(norm_layout)
        scroll_layout.addWidget(norm_group)

        action_group = QGroupBox("执行预处理")
        action_layout = QHBoxLayout()

        self.btn_preview = QPushButton("预览处理结果")
        self.btn_preview.clicked.connect(self.preview_result)
        action_layout.addWidget(self.btn_preview)

        self.btn_apply = QPushButton("应用预处理")
        self.btn_apply.clicked.connect(self.apply_preprocess)
        action_layout.addWidget(self.btn_apply)

        action_layout.addWidget(QWidget(), 1)
        action_group.setLayout(action_layout)
        scroll_layout.addWidget(action_group)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

    def set_data(self, data_info):
        self.data_info = data_info

    def analyze_data(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            QMessageBox.warning(self, "警告", "请先在数据导入页面加载数据")
            return

        self.auto_analysis = self.preprocessor.auto_analyze(self.current_df)
        result = "列名 | 缺失率% | 建议缺失值处理 | 是否有异常值\n"
        result += "=" * 70 + "\n"
        for col, info in self.auto_analysis.items():
            result += f"{col[:15]:15} | {info['missing_rate']:7.1f} | {info['suggested_missing_method']:12} | {'是' if info['has_outliers'] else '否'}\n"
        self.analysis_text.setPlainText(result)

    def detect_outliers(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            QMessageBox.warning(self, "警告", "请先在数据导入页面加载数据")
            return

        method = self.cb_outlier_method.currentText().lower().replace("-", "")
        threshold = self.sb_outlier_threshold.value()
        outliers = self.preprocessor.detect_outliers(self.current_df, method, threshold)

        result = ""
        total = 0
        for col, indices in outliers.items():
            if len(indices) > 0:
                result += f"{col}: {len(indices)} 个异常值\n"
                total += len(indices)
        result += f"\n共检测到 {total} 个异常值"
        self.outlier_text.setPlainText(result)
        self.detected_outliers = outliers

    def preview_result(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return

        preview_df = self._apply_preprocessing(self.current_df.copy())
        self.table_preview = QTextEdit()
        self.table_preview.setReadOnly(True)
        info = f"预处理后数据形状: {preview_df.shape}\n"
        info += f"缺失值数量: {preview_df.isna().sum().sum()}\n"
        info += f"数据预览(前5行):\n{preview_df.head()}"
        QMessageBox.information(self, "预览", info)

    def _apply_preprocessing(self, df):
        missing_method = self.cb_missing_method.currentText()
        method_map = {"自动": "auto", "均值": "mean", "中位数": "median",
                     "前向填充": "forward_fill", "后向填充": "backward_fill",
                     "插值": "interpolate", "KNN填充": "knn"}

        if missing_method != "自动":
            df = self.preprocessor.handle_missing_values(df, method_map[missing_method])
        else:
            df = self.preprocessor.handle_missing_values(df, "auto", self.auto_analysis)

        outlier_method = self.cb_outlier_method.currentText().lower()
        threshold = self.sb_outlier_threshold.value()
        outliers = self.preprocessor.detect_outliers(df, outlier_method, threshold)
        outlier_handle = self.cb_outlier_handle.currentText()
        df = self.preprocessor.handle_outliers(df, outlier_handle.lower(), outliers)

        norm_method = self.cb_norm_method.currentText()
        if "不归一化" not in norm_method:
            if "Min-Max" in norm_method:
                df, _ = self.preprocessor.normalize(df, "minmax",
                                                    (self.sb_norm_min.value(), self.sb_norm_max.value()))
            else:
                df, _ = self.preprocessor.normalize(df, "standard")

        return df

    def apply_preprocess(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return

        self.preprocessed_df = self._apply_preprocessing(self.current_df.copy())

        result = {
            "df": self.preprocessed_df,
            "preprocessor": self.preprocessor,
            "original_shape": self.current_df.shape,
            "processed_shape": self.preprocessed_df.shape
        }

        self.preprocess_complete.emit(result)
        QMessageBox.information(self, "成功", "预处理完成，数据已准备好进行模型配置")