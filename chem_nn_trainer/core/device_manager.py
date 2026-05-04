# ┌────────────────────────────────────────────────────────────┐
# │  硬件设备管理器                                            │
# │  检测可用计算设备（CPU/GPU）并获取设备信息                  │
# └────────────────────────────────────────────────────────────┘

import torch
import platform
from typing import Dict, List, Optional


class DeviceManager:
    """硬件设备管理器

    功能：
    - 自动检测 CUDA GPU 是否可用
    - 获取 GPU 详细信息（名称、显存、CUDA 版本等）
    - 提供设备选择与显存限制接口
    """

    @staticmethod
    def get_available_devices() -> Dict:
        """获取所有可用计算设备列表

        Returns:
            dict: {
                "cpu": True,
                "cuda": bool,          # CUDA GPU 是否可用
                "cuda_info": dict or None
            }
        """
        devices = {
            "cpu": True,
            "cuda": False,
            "cuda_info": None
        }

        if torch.cuda.is_available():
            devices["cuda"] = True
            devices["cuda_info"] = DeviceManager._get_cuda_info()

        return devices

    @staticmethod
    def _get_cuda_info() -> Dict:
        """获取 CUDA GPU 详细信息"""
        cuda_info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,   # 已分配显存 (GB)
            "memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,      # 已预留显存 (GB)
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "platform": platform.system()
        }
        return cuda_info

    @staticmethod
    def get_device(device_type: str = "auto") -> torch.device:
        """根据类型字符串获取 PyTorch 设备对象

        Args:
            device_type: "auto"(自动选择 GPU > CPU) / "cuda" / "cpu"

        Returns:
            torch.device
        """
        if device_type == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        elif device_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    @staticmethod
    def set_device_memory_limit(device_id: int = 0, fraction: float = 0.8):
        """设置 CUDA 显存使用上限（占总显存比例）

        Args:
            device_id: GPU 设备 ID
            fraction: 允许 PyTorch 使用的显存比例 (0~1)
        """
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction, device_id)

    @staticmethod
    def get_device_name(device: torch.device) -> str:
        """获取设备名称字符串"""
        if device.type == "cuda":
            return torch.cuda.get_device_name(device)
        return "CPU"

    @staticmethod
    def check_cuda_available() -> bool:
        """检查 CUDA 是否可用"""
        return torch.cuda.is_available()

    @staticmethod
    def get_memory_info(device: Optional[torch.device] = None) -> Dict:
        """获取指定设备的显存使用信息

        Returns:
            dict: {"allocated": GB, "reserved": GB, "max_allocated": GB}
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(device) / 1024**3,
                "reserved": torch.cuda.memory_reserved(device) / 1024**3,
                "max_allocated": torch.cuda.max_memory_allocated(device) / 1024**3
            }
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
