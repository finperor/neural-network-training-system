import torch
import torch.nn as nn
from typing import List, Dict


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 hidden_layers: List[int] = None,
                 activation: str = "relu",
                 dropout: float = 0.2):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.net(x)


class RNNModel(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 hidden_size: int = 64, num_layers: int = 2,
                 bidirectional: bool = False, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidirectional,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 hidden_size: int = 64, num_layers: int = 2,
                 bidirectional: bool = False, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class GRUModel(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 hidden_size: int = 64, num_layers: int = 2,
                 bidirectional: bool = False, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidirectional,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


class CNN1DModel(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 hidden_channels: List[int] = None,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64]

        layers = []
        prev_channels = 1

        for i, channels in enumerate(hidden_channels):
            layers.append(nn.Conv1d(prev_channels, channels, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            prev_channels = channels

        self.conv = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Sequential(
            nn.Linear(prev_channels * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(2) != 1 and x.size(1) != 1:
            x = x.transpose(1, 2)
        out = self.conv(x)
        out = self.pool(out)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)


class ModelBuilder:
    @staticmethod
    def build(model_type: str, input_size: int, output_size: int,
              extra_config: Dict = None) -> nn.Module:
        if extra_config is None:
            extra_config = {}

        model_type = model_type.lower()

        if model_type == "mlp":
            return MLP(
                input_size=input_size,
                output_size=output_size,
                hidden_layers=extra_config.get("hidden_layers", [64, 32]),
                activation=extra_config.get("activation", "relu"),
                dropout=extra_config.get("dropout", 0.2)
            )
        elif model_type == "rnn":
            return RNNModel(
                input_size=input_size,
                output_size=output_size,
                hidden_size=extra_config.get("hidden_size", 64),
                num_layers=extra_config.get("num_layers", 2),
                bidirectional=extra_config.get("bidirectional", False),
                dropout=extra_config.get("dropout", 0.2)
            )
        elif model_type == "lstm":
            return LSTMModel(
                input_size=input_size,
                output_size=output_size,
                hidden_size=extra_config.get("hidden_size", 64),
                num_layers=extra_config.get("num_layers", 2),
                bidirectional=extra_config.get("bidirectional", False),
                dropout=extra_config.get("dropout", 0.2)
            )
        elif model_type == "gru":
            return GRUModel(
                input_size=input_size,
                output_size=output_size,
                hidden_size=extra_config.get("hidden_size", 64),
                num_layers=extra_config.get("num_layers", 2),
                bidirectional=extra_config.get("bidirectional", False),
                dropout=extra_config.get("dropout", 0.2)
            )
        elif model_type in ("cnn-1d", "cnn1d"):
            return CNN1DModel(
                input_size=input_size,
                output_size=output_size,
                hidden_channels=extra_config.get("hidden_channels", [32, 64]),
                kernel_size=extra_config.get("kernel_size", 3),
                dropout=extra_config.get("dropout", 0.2)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
