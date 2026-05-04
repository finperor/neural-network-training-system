# 化工过程神经网络训练系统 - 技术文档

## 目录

1. [系统概述](#1-系统概述)
2. [项目架构](#2-项目架构)
3. [数据模块详解](#3-数据模块详解)
4. [模型模块详解](#4-模型模块详解)
5. [核心训练模块详解](#5-核心训练模块详解)
6. [用户界面模块详解](#6-用户界面模块详解)
7. [关键技术实现](#7-关键技术实现)
8. [扩展与优化](#8-扩展与优化)

---

## 1. 系统概述

### 1.1 设计目标

本系统旨在为化工过程工程师提供一个简洁易用的神经网络训练工具，用于基于历史运行数据建立预测模型。典型应用场景包括：

- 反应器温度预测
- 产品质量预测
- 工艺参数优化
- 异常工况检测

### 1.2 技术栈

| 层次 | 技术选择 | 理由 |
|------|----------|------|
| 深度学习 | PyTorch | 成熟的深度学习框架，GPU支持完善 |
| 数据处理 | Pandas/NumPy | 工业数据处理标准库 |
| 桌面UI | PyQt6 | Python桌面应用开发首选 |
| 可视化 | Matplotlib | 灵活的绑图能力 |

### 1.3 系统流程

```
┌──────────┐   ┌─────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ CSV数据  │──→│ 数据预处理  │──→│ 模型配置 │──→│ 模型训练 │──→│ 结果分析 │
└──────────┘   └─────────────┘   └──────────┘   └──────────┘   └──────────┘
```

---

## 2. 项目架构

### 2.1 目录结构

```
chem_nn_trainer/
├── main.py              # 程序入口
├── requirements.txt     # 依赖清单
├── core/                # 核心功能
│   ├── trainer.py       # 训练控制器
│   └── device_manager.py # 硬件管理
├── data/                # 数据处理
│   ├── data_loader.py   # 数据加载
│   ├── preprocessor.py  # 数据预处理
│   └── dataset.py       # 数据集封装
├── models/              # 神经网络
│   ├── nn_builder.py    # 网络构建
│   └── evaluator.py     # 模型评估
└── ui/                  # 图形界面
    ├── main_window.py   # 主窗口
    ├── data_panel.py    # 数据导入页
    ├── preprocess_panel.py # 数据预处理页
    ├── model_panel.py   # 模型配置页
    ├── train_panel.py   # 训练监控页
    └── result_panel.py  # 结果分析页
```

### 2.2 模块依赖关系

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│                   (程序入口、创建主窗口)                  │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌─────────┐
    │  data/  │    │ models/  │    │  core/  │
    │ (数据层)│    │ (模型层) │    │ (训练层)│
    └────┬────┘    └────┬─────┘    └────┬────┘
         │              │               │
         └──────────────┼───────────────┘
                        ▼
               ┌─────────────────┐
               │      ui/        │
               │  (用户界面层)     │
               └─────────────────┘
```

---

## 3. 数据模块详解

### 3.1 DataLoader - 数据加载

**文件**: `data/data_loader.py`

**职责**: 负责从文件系统加载CSV数据，并提供基本统计信息

**核心类**: `DataLoader`

```python
class DataLoader:
    def load_csv(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """加载CSV文件，返回数据和基本信息"""
        pass

    def get_column_stats(self, column: str) -> Dict:
        """获取单列统计信息"""
        pass
```

**实现方法**:

| 方法 | 功能 | 返回值 |
|------|------|--------|
| `load_csv()` | 使用Pandas读取CSV | DataFrame, 行/列数, 列名等 |
| `get_column_stats()` | 计算列的均值/标准差/缺失率 | Dict |

**数据格式要求**:
- 第一行为列名（变量名）
- 后续每行为一个时间点的采样数据
- 支持数值型和字符串型数据

### 3.2 DataPreprocessor - 数据预处理

**文件**: `data/preprocessor.py`

**职责**: 处理缺失值、异常值，提供归一化功能

**核心功能**:

#### 3.2.1 缺失值处理

**支持的方法**:

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| `mean` | 均值填充 | 缺失率<5%，数据分布均匀 |
| `median` | 中位数填充 | 存在极端值 |
| `forward_fill` | 前向填充 | 时序数据 |
| `backward_fill` | 后向填充 | 时序数据 |
| `interpolate` | 线性插值 | 连续变化的过程变量 |
| `knn` | K近邻填充 | 缺失率较高，有相关特征 |

**自动选择算法**:
```python
def _suggest_missing_method(self, missing_rate: float) -> str:
    if missing_rate == 0:      return "none"
    elif missing_rate < 5:     return "mean"
    elif missing_rate < 20:    return "interpolate"
    elif missing_rate < 50:    return "knn"
    else:                      return "forward_fill"
```

#### 3.2.2 异常值检测

**支持的算法**:

1. **IQR方法** (四分位距)
   - 计算 Q1(25%分位) 和 Q3(75%分位)
   - IQR = Q3 - Q1
   - 异常值定义: value < Q1 - 1.5×IQR 或 value > Q3 + 1.5×IQR

2. **Z-Score方法**
   - z = (x - μ) / σ
   - 异常值定义: |z| > threshold (默认3)

#### 3.2.3 数据归一化

| 方法 | 公式 | 适用场景 |
|------|------|----------|
| Min-Max | x' = (x - min) / (max - min) | 数据有确定范围 |
| Z-Score | x' = (x - μ) / σ | 数据近似正态分布 |

### 3.3 DatasetSplitter - 数据集划分

**文件**: `data/dataset.py`

**职责**: 将数据划分为训练集/验证集/测试集

**划分方式**:

1. **随机划分** (适用于非时序数据)
   - 使用sklearn的train_test_split
   - 两次划分: 训练vs(验证+测试)，再细分验证和测试

2. **时序划分** (适用于时序数据)
   - 按时间顺序划分
   - 训练集在前，验证集居中，测试集在后
   - 避免数据穿越（future leakage）

**PyTorch数据加载**:
```python
class SimpleDataset(Dataset):
    """简单的特征-标签数据集"""
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

class TimeSeriesDataset(Dataset):
    """时序数据集，用于RNN/LSTM/GRU"""
    def __init__(self, data: np.ndarray, target_indices: List[int], seq_length: int):
        # 将多个时间步的数据作为输入，下一个时间步的目标作为输出
        pass
```

---

## 4. 模型模块详解

### 4.1 ModelBuilder - 神经网络构建

**文件**: `models/nn_builder.py`

**职责**: 根据用户配置动态构建神经网络

**支持的模型**:

#### 4.1.1 MLP (全连接神经网络)

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size,
                 activation="relu", dropout=0.0):
        # 层级结构: input -> hidden1 -> hidden2 -> ... -> output
        # 每层: Linear -> Activation -> Dropout(可选)
        pass
```

**适用场景**: 数据量较小，特征与目标关系复杂

#### 4.1.2 RNN (循环神经网络)

```python
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout=0.0, bidirectional=False):
        # 层级结构: input -> RNN -> Dense -> output
        # bidirectional=True 可捕获双向依赖
        pass
```

**适用场景**: 短时序依赖，序列长度<50

#### 4.1.3 LSTM (长短期记忆网络)

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout=0.0, bidirectional=False):
        # 使用LSTM单元解决梯度消失问题
        # 可捕获较长时序依赖
        pass
```

**适用场景**: 较长时序依赖，化工过程时序预测首选

#### 4.1.4 GRU (门控循环单元)

```python
class GRUModel(nn.Module):
    # LSTM的简化版本，参数更少，训练更快
    pass
```

**适用场景**: 与LSTM类似，训练资源有限时

#### 4.1.5 CNN-1D (一维卷积)

```python
class CNN1DModel(nn.Module):
    def __init__(self, input_size, hidden_channels, kernel_size,
                 output_size, activation="relu", dropout=0.0):
        # 使用一维卷积提取局部特征
        pass
```

**适用场景**: 需要局部模式检测的时序数据

**网络结构确定规则**:

| 层次 | 节点数确定方式 |
|------|----------------|
| 输入层 | 输入特征数 = 总列数 - 目标列数 |
| 隐藏层 | 用户自定义，每层可不同 |
| 输出层 | 目标变量数量（回归=1，分类=类别数） |

### 4.2 ModelEvaluator - 模型评估

**文件**: `models/evaluator.py`

**职责**: 计算模型性能指标

**回归任务指标**:

| 指标 | 公式 | 含义 |
|------|------|------|
| MSE | (1/n)Σ(y-ŷ)² | 均方误差，越小越好 |
| RMSE | √MSE | 均方根误差，与原数据同单位 |
| MAE | (1/n)Σ|y-ŷ| | 平均绝对误差，对异常值不敏感 |
| R² | 1 - SS_res/SS_tot | 决定系数，越接近1越好 |
| MAPE | (1/n)Σ|y-ŷ|/|y|×100% | 平均绝对百分比误差 |

**分类任务指标**:

| 指标 | 含义 |
|------|------|
| Accuracy | 正确率 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1 | 精确率和召回率的调和平均 |

---

## 5. 核心训练模块详解

### 5.1 Trainer - 训练控制器

**文件**: `core/trainer.py`

**职责**: 管理训练流程，实现反向传播、参数更新、早停等

**核心功能**:

#### 5.1.1 优化器选择

| 优化器 | 特点 | 适用场景 |
|--------|------|----------|
| Adam | 自适应学习率，收敛快 | 默认首选 |
| SGD | 简单，可能陷入局部最优 | 大数据量 |
| RMSprop | 自适应学习率 | 递归神经网络 |
| AdamW | Adam+L2正则 | 需要正则化时 |

#### 5.1.2 学习率调度

| 调度器 | 策略 | 参数 |
|--------|------|------|
| StepLR | 每N轮降低学习率 | step_size, gamma |
| ExponentialLR | 指数衰减 | gamma |
| CosineAnnealing | 余弦退火 | T_max |
| ReduceLROnPlateau | 验证损失不下降时降低 | patience, factor |

#### 5.1.3 早停机制

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model_state = model.state_dict()
    no_improve_count = 0
else:
    no_improve_count += 1
    if no_improve_count >= patience:
        # 停止训练
        break
```

**作用**: 防止过拟合，当验证损失连续N轮不下降时停止

#### 5.1.4 梯度裁剪

```python
if gradient_clip:
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
```

**作用**: 防止梯度爆炸，常用于RNN/LSTM训练

### 5.2 DeviceManager - 硬件管理

**文件**: `core/device_manager.py`

**职责**: 检测和管理计算设备

**功能**:

```python
class DeviceManager:
    @staticmethod
    def get_available_devices() -> Dict:
        # 检测CPU和GPU支持
        # 返回设备信息字典

    @staticmethod
    def get_device(device_type: str = "auto") -> torch.device:
        # "auto": 自动选择（有GPU用GPU，否则CPU）
        # "cuda": 强制使用GPU
        # "cpu": 强制使用CPU
```

**CUDA检测**:
```python
if torch.cuda.is_available():
    device_info = {
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,
        "cuda_version": torch.version.cuda
    }
```

---

## 6. 用户界面模块详解

### 6.1 MainWindow - 主窗口

**文件**: `ui/main_window.py`

**职责**: 整合所有面板，管理页面切换和数据传递

**页面切换**: 使用QTabWidget实现左侧标签页

**信号槽机制**:

| 信号 | 发送者 | 接收者 | 作用 |
|------|--------|--------|------|
| data_loaded | DataPanel | MainWindow | 传递原始数据 |
| preprocess_complete | PreprocessPanel | MainWindow | 传递预处理后数据 |
| model_ready | ModelPanel | MainWindow | 传递模型配置 |
| training_complete | TrainPanel | MainWindow | 传递训练结果 |

### 6.2 DataPanel - 数据导入

**功能**:
- 加载CSV文件
- 数据预览（表格显示前20行）
- 数据统计（列的缺失率、均值、标准差等）

**关键组件**:
- QPushButton: "加载CSV"、"数据预览"、"数据统计"
- QTableWidget: 表格显示
- QTextEdit: 统计信息显示

### 6.3 PreprocessPanel - 数据预处理

**功能**:
- 缺失值处理方法选择
- 异常值检测和处理
- 数据归一化

**自动分析**:
```python
def auto_analyze(self, df: pd.DataFrame) -> Dict:
    # 分析每列的缺失率和异常值
    # 返回每列推荐的预处理方法
```

### 6.4 ModelPanel - 模型配置

**功能**:
- 选择目标变量（要预测的列）
- 选择任务类型（回归/分类）
- 选择模型架构
- 配置网络结构（隐藏层数、节点数）
- 配置训练参数

**网络结构可视化**:
```python
def on_model_type_changed(self, text):
    # 根据模型类型显示不同的配置界面
    # MLP: 显示隐藏层列表，每层可设置节点数
    # RNN/LSTM/GRU: 显示隐藏单元数、层数、双向选项
    # CNN-1D: 显示卷积通道、核大小
```

### 6.5 TrainPanel - 训练监控

**功能**:
- 选择计算设备（CPU/GPU）
- 配置数据集划分比例
- 启动/暂停/停止训练
- 实时显示训练进度和损失曲线

**实时绘图**:
```python
class TrainingMonitor(QWidget):
    def paintEvent(self, event):
        # 使用QPainter绘制损失曲线
        # 绿色: 训练损失
        # 红色: 验证损失
```

### 6.6 ResultPanel - 结果分析

**功能**:
- 显示评估指标
- 绘制预测对比图
- 绘制真实值vs预测值散点图
- 导出模型和结果

**可视化组件**:
- PredictionPlot: 时序预测曲线对比
- ScatterPlot: 散点图（理想情况下应落在对角线上）

---

## 7. 关键技术实现

### 7.1 时序数据处理

对于RNN/LSTM/GRU模型，需要将数据转换为序列格式：

```python
class TimeSeriesDataset(Dataset):
    def __getitem__(self, idx):
        # 输入: seq_length个时间步的特征
        x = self.data[idx:idx + self.seq_length]
        # 输出: 下一个时间步的目标值
        y = self.data[idx + self.seq_length, self.target_indices]
        return x, y
```

### 7.2 动态网络构建

根据用户配置动态创建网络：

```python
def build(model_type: str, input_size: int, output_size: int, config: Dict):
    if model_type == "mlp":
        return MLP(input_size, config["hidden_layers"], output_size, ...)
    elif model_type == "lstm":
        return LSTMModel(input_size, config["hidden_size"], config["num_layers"], ...)
```

### 7.3 GPU/CPU无缝切换

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch_x = batch_x.to(device)  # 数据也要移到同一设备
```

### 7.4 训练过程监控

使用QTimer实现定时更新UI：

```python
self.timer = QTimer()
self.timer.timeout.connect(self.update_training)
self.timer.start(500)  # 每500ms更新一次
```

---

## 8. 扩展与优化

### 8.1 后续可添加的功能

| 功能 | 描述 | 优先级 |
|------|------|--------|
| 模型对比 | 多个模型对比 | 高 |
| 超参数搜索 | 自动寻找最优超参数 | 中 |
| 特征重要性 | SHAP分析 | 中 |
| 批量预测 | 新数据批量预测 | 中 |
| 分布式训练 | 多GPU训练 | 低 |
| 自定义网络层 | 化工领域特定层 | 低 |

### 8.2 性能优化建议

1. **数据预处理优化**: 使用Numba加速数值计算
2. **训练加速**: 使用DataLoader的num_workers多进程加载
3. **内存优化**: 处理大数据时使用内存映射文件

### 8.3 部署注意事项

1. Windows部署时建议使用PyInstaller打包
2. 包含所有依赖的standalone版本
3. 注意PyTorch的CUDA版本兼容性

---

## 附录：代码调用流程

```
main.py
  └── MainWindow
        ├── DataPanel.load_file()
        │     └── DataLoader.load_csv()
        │
        ├── PreprocessPanel.apply_preprocess()
        │     └── DataPreprocessor.handle_missing_values()
        │     └── DataPreprocessor.detect_outliers()
        │     └── DataPreprocessor.normalize()
        │
        ├── ModelPanel.confirm_config()
        │     └── ModelBuilder.build()
        │
        ├── TrainPanel.start_training()
        │     └── DatasetSplitter.split()
        │     └── Trainer.train()
        │     └── ModelEvaluator.evaluate()
        │
        └── ResultPanel.set_results()
              └── 模型导出
```

---

*文档版本: 1.0*
*更新时间: 2025*