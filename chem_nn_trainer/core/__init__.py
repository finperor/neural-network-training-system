# ┌────────────────────────────────────────────────────────────┐
# │  core 包 - 核心训练引擎与设备管理                           │
# └────────────────────────────────────────────────────────────┘

from .trainer import Trainer
from .device_manager import DeviceManager

__all__ = ["Trainer", "DeviceManager"]
