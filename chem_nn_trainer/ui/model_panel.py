from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
                             QCheckBox, QTextEdit, QListWidget, QListWidgetItem,
                             QMessageBox, QScrollArea, QSplitter, QTableWidget,
                             QTableWidgetItem)
from PyQt6.QtCore import pyqtSignal, Qt
import pandas as pd


class ModelPanel(QWidget):
    model_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.preprocessed_data = None
        self.feature_columns = []
        self.target_columns = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        target_group = QGroupBox("目标变量选择")
        target_layout = QVBoxLayout()
        target_layout.addWidget(QLabel("选择要预测的目标列:"))
        self.target_list = QListWidget()
        self.target_list.setMaximumHeight(120)
        target_layout.addWidget(self.target_list)
        target_group.setLayout(target_layout)
        left_layout.addWidget(target_group)

        type_group = QGroupBox("任务类型")
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("任务类型:"))
        self.cb_task_type = QComboBox()
        self.cb_task_type.addItems(["回归", "分类"])
        type_layout.addWidget(self.cb_task_type)
        type_layout.addWidget(QWidget(), 1)
        type_group.setLayout(type_layout)
        left_layout.addWidget(type_group)

        splitter.addWidget(left_widget)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)

        model_type_group = QGroupBox("神经网络模型")
        model_type_layout = QVBoxLayout()

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型类型:"))
        self.cb_model_type = QComboBox()
        self.cb_model_type.addItems(["MLP (全连接)", "RNN (循环神经网络)", "LSTM", "GRU", "CNN-1D"])
        self.cb_model_type.currentTextChanged.connect(self.on_model_type_changed)
        model_row.addWidget(self.cb_model_type)
        model_type_layout.addLayout(model_row)

        model_type_group.setLayout(model_type_layout)
        center_layout.addWidget(model_type_group)

        self.config_group = QGroupBox("网络结构配置")
        self.config_layout = QVBoxLayout()
        self.setup_mlp_config()
        self.config_group.setLayout(self.config_layout)
        center_layout.addWidget(self.config_group)

        splitter.addWidget(center_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        training_group = QGroupBox("训练参数")
        train_layout = QVBoxLayout()

        optimizer_row = QHBoxLayout()
        optimizer_row.addWidget(QLabel("优化器:"))
        self.cb_optimizer = QComboBox()
        self.cb_optimizer.addItems(["Adam", "SGD", "RMSprop", "AdamW"])
        optimizer_row.addWidget(self.cb_optimizer)
        train_layout.addLayout(optimizer_row)

        lr_row = QHBoxLayout()
        lr_row.addWidget(QLabel("学习率:"))
        self.sb_lr = QDoubleSpinBox()
        self.sb_lr.setRange(0.00001, 0.1)
        self.sb_lr.setValue(0.001)
        self.sb_lr.setDecimals(5)
        self.sb_lr.setStepType(QSpinBox.StepType.AdaptiveDecimalStepType)
        lr_row.addWidget(self.sb_lr)
        train_layout.addLayout(lr_row)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("批次大小:"))
        self.sb_batch = QSpinBox()
        self.sb_batch.setRange(4, 512)
        self.sb_batch.setValue(32)
        self.sb_batch.setSingleStep(4)
        batch_row.addWidget(self.sb_batch)
        train_layout.addLayout(batch_row)

        epoch_row = QHBoxLayout()
        epoch_row.addWidget(QLabel("训练轮次:"))
        self.sb_epochs = QSpinBox()
        self.sb_epochs.setRange(1, 1000)
        self.sb_epochs.setValue(100)
        epoch_row.addWidget(self.sb_epochs)
        train_layout.addLayout(epoch_row)

        loss_row = QHBoxLayout()
        loss_row.addWidget(QLabel("损失函数:"))
        self.cb_loss = QComboBox()
        self.cb_loss.addItems(["MSE", "MAE", "Huber"])
        loss_row.addWidget(self.cb_loss)
        train_layout.addLayout(loss_row)

        scheduler_row = QHBoxLayout()
        scheduler_row.addWidget(QLabel("学习率调度:"))
        self.cb_scheduler = QComboBox()
        self.cb_scheduler.addItems(["无", "阶梯衰减", "指数衰减", "余弦退火", "ReduceLROnPlateau"])
        scheduler_row.addWidget(self.cb_scheduler)
        train_layout.addLayout(scheduler_row)

        regular_row = QHBoxLayout()
        regular_row.addWidget(QLabel("正则化:"))
        self.cb_regularization = QComboBox()
        self.cb_regularization.addItems(["无", "L2正则化"])
        regular_row.addWidget(self.cb_regularization)
        self.sb_reg_lambda = QDoubleSpinBox()
        self.sb_reg_lambda.setRange(0.0001, 1.0)
        self.sb_reg_lambda.setValue(0.01)
        self.sb_reg_lambda.setDecimals(4)
        regular_row.addWidget(self.sb_reg_lambda)
        train_layout.addLayout(regular_row)

        dropout_row = QHBoxLayout()
        dropout_row.addWidget(QLabel("Dropout:"))
        self.sb_dropout = QDoubleSpinBox()
        self.sb_dropout.setRange(0, 0.9)
        self.sb_dropout.setValue(0.2)
        self.sb_dropout.setSingleStep(0.05)
        dropout_row.addWidget(self.sb_dropout)
        train_layout.addLayout(dropout_row)

        early_stop_row = QHBoxLayout()
        early_stop_row.addWidget(QLabel("早停耐心:"))
        self.sb_patience = QSpinBox()
        self.sb_patience.setRange(1, 50)
        self.sb_patience.setValue(10)
        early_stop_row.addWidget(self.sb_patience)
        early_stop_row.addWidget(QLabel("轮"))
        train_layout.addLayout(early_stop_row)

        training_group.setLayout(train_layout)
        right_layout.addWidget(training_group)

        splitter.addWidget(right_widget)

        layout.addWidget(splitter)

        bottom_layout = QHBoxLayout()
        self.btn_preview = QPushButton("预览模型结构")
        self.btn_preview.clicked.connect(self.preview_model)
        bottom_layout.addWidget(self.btn_preview)

        self.btn_confirm = QPushButton("确认配置")
        self.btn_confirm.clicked.connect(self.confirm_config)
        bottom_layout.addWidget(self.btn_confirm)
        bottom_layout.addWidget(QWidget(), 1)

        layout.addLayout(bottom_layout)

    def setup_mlp_config(self):
        self.layer_config = []
        self.layer_spinboxes = []

        row = QHBoxLayout()
        row.addWidget(QLabel("隐藏层配置:"))
        btn_add = QPushButton("添加层")
        btn_add.clicked.connect(self.add_layer)
        row.addWidget(btn_add)
        btn_remove = QPushButton("移除层")
        btn_remove.clicked.connect(self.remove_layer)
        row.addWidget(btn_remove)
        self.config_layout.addLayout(row)

    def clear_config_layout(self):
        while self.config_layout.count():
            item = self.config_layout.takeAt(0)
            w = item.widget()
            if w:
                w.hide()
                w.deleteLater()
            sub = item.layout()
            if sub:
                while sub.count():
                    sub_item = sub.takeAt(0)
                    sw = sub_item.widget()
                    if sw:
                        sw.hide()
                        sw.deleteLater()
                    ssub = sub_item.layout()
                    if ssub:
                        while ssub.count():
                            ss_item = ssub.takeAt(0)
                            ssw = ss_item.widget()
                            if ssw:
                                ssw.hide()
                                ssw.deleteLater()

    def on_model_type_changed(self, text):
        self.clear_config_layout()

        model_type = text.split()[0].lower()

        if model_type == "mlp":
            self.setup_mlp_config()
        elif model_type in ["rnn", "lstm", "gru"]:
            self.setup_rnn_config(model_type)
        elif model_type == "cnn-1d":
            self.setup_cnn_config()

    def setup_rnn_config(self, model_type):
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{model_type.upper()} 隐藏单元:"))
        self.sb_hidden = QSpinBox()
        self.sb_hidden.setRange(8, 512)
        self.sb_hidden.setValue(64)
        self.sb_hidden.setSingleStep(8)
        row.addWidget(self.sb_hidden)
        row.addWidget(QLabel("层数:"))
        self.sb_num_layers = QSpinBox()
        self.sb_num_layers.setRange(1, 5)
        self.sb_num_layers.setValue(2)
        row.addWidget(self.sb_num_layers)
        row.addWidget(QLabel("双向:"))
        self.cb_bidirectional = QCheckBox()
        self.cb_bidirectional.setChecked(False)
        row.addWidget(self.cb_bidirectional)
        self.config_layout.addLayout(row)

    def setup_cnn_config(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("卷积通道:"))
        self.sb_channels1 = QSpinBox()
        self.sb_channels1.setRange(8, 256)
        self.sb_channels1.setValue(32)
        row.addWidget(self.sb_channels1)
        self.sb_channels2 = QSpinBox()
        self.sb_channels2.setRange(8, 256)
        self.sb_channels2.setValue(64)
        row.addWidget(self.sb_channels2)
        row.addWidget(QLabel("卷积核大小:"))
        self.sb_kernel = QSpinBox()
        self.sb_kernel.setRange(2, 7)
        self.sb_kernel.setValue(3)
        row.addWidget(self.sb_kernel)
        self.config_layout.addLayout(row)

    def add_layer(self):
        row = QHBoxLayout()
        label = QLabel(f"隐藏层 {len(self.layer_spinboxes) + 1}:")
        sb = QSpinBox()
        sb.setRange(4, 512)
        sb.setValue(64)
        sb.setSingleStep(8)
        self.layer_spinboxes.append(sb)
        row.addWidget(label)
        row.addWidget(sb)
        self.config_layout.addLayout(row)

    def remove_layer(self):
        if self.layer_spinboxes:
            self.layer_spinboxes.pop()
            if self.config_layout.count() > 2:
                item = self.config_layout.takeAt(self.config_layout.count() - 1)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    item.layout().deleteLater()

    def set_data(self, data_info):
        pass

    def set_preprocessed_data(self, data):
        self.preprocessed_data = data
        df = data["df"]
        columns = list(df.columns)
        self.target_list.clear()
        for col in columns:
            item = QListWidgetItem(col)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.target_list.addItem(item)
        self.feature_columns = [col for col in columns]

    def preview_model(self):
        if not self.preprocessed_data:
            QMessageBox.warning(self, "警告", "请先完成数据预处理")
            return

        targets = self.get_selected_targets()
        if not targets:
            QMessageBox.warning(self, "警告", "请至少选择一个目标变量")
            return

        config = self.build_config()
        preview = f"模型预览:\n"
        preview += f"模型类型: {self.cb_model_type.currentText()}\n"
        preview += f"输入特征: {len(self.feature_columns) - len(targets)}\n"
        preview += f"输出目标: {len(targets)}\n"
        preview += f"\n网络结构:\n"
        preview += f"  输入层: {len(self.feature_columns) - len(targets)} 节点\n"

        if self.cb_model_type.currentText().startswith("MLP"):
            for i, sb in enumerate(self.layer_spinboxes):
                preview += f"  隐藏层 {i+1}: {sb.value()} 节点 ({self.cb_optimizer.currentText()}, ReLU, Dropout={self.sb_dropout.value()})\n"
        else:
            preview += f"  隐藏层: {self.sb_hidden.value()} 节点, {self.sb_num_layers.value()} 层\n"

        preview += f"  输出层: {len(targets)} 节点\n"
        preview += f"\n训练参数:\n"
        preview += f"  优化器: {self.cb_optimizer.currentText()}\n"
        preview += f"  学习率: {self.sb_lr.value()}\n"
        preview += f"  批次大小: {self.sb_batch.value()}\n"
        preview += f"  训练轮次: {self.sb_epochs.value()}\n"
        preview += f"  早停耐心: {self.sb_patience.value()} 轮"

        QMessageBox.information(self, "模型预览", preview)

    def get_selected_targets(self):
        targets = []
        for i in range(self.target_list.count()):
            item = self.target_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                targets.append(item.text())
        return targets

    def build_config(self):
        targets = self.get_selected_targets()
        model_type = self.cb_model_type.currentText().split()[0].lower()

        config = {
            "task_type": self.cb_task_type.currentText(),
            "model_type": model_type,
            "input_size": len(self.feature_columns) - len(targets),
            "output_size": len(targets),
            "target_columns": targets,
            "optimizer": self.cb_optimizer.currentText().lower(),
            "learning_rate": self.sb_lr.value(),
            "batch_size": self.sb_batch.value(),
            "epochs": self.sb_epochs.value(),
            "loss_fn": self.cb_loss.currentText().lower(),
            "scheduler": self.cb_scheduler.currentText(),
            "dropout": self.sb_dropout.value(),
            "patience": self.sb_patience.value(),
            "regularization": self.cb_regularization.currentText(),
            "reg_lambda": self.sb_reg_lambda.value(),
            "data": self.preprocessed_data
        }

        if model_type == "mlp":
            config["hidden_layers"] = [sb.value() for sb in self.layer_spinboxes]
            config["activation"] = "relu"
        elif model_type in ["rnn", "lstm", "gru"]:
            config["hidden_size"] = self.sb_hidden.value()
            config["num_layers"] = self.sb_num_layers.value()
            config["bidirectional"] = self.cb_bidirectional.isChecked()
        elif model_type == "cnn-1d":
            config["hidden_channels"] = [self.sb_channels1.value(), self.sb_channels2.value()]
            config["kernel_size"] = self.sb_kernel.value()

        return config

    def confirm_config(self):
        if not self.preprocessed_data:
            QMessageBox.warning(self, "警告", "请先完成数据预处理")
            return

        targets = self.get_selected_targets()
        if not targets:
            QMessageBox.warning(self, "警告", "请至少选择一个目标变量")
            return

        config = self.build_config()
        self.model_ready.emit(config)
        QMessageBox.information(self, "成功", "模型配置已确认，跳转到训练页面")