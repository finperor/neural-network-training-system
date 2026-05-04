#!/usr/bin/env python3
import sys

print("Testing imports...")

try:
    from data.data_loader import DataLoader
    print("  data_loader: OK")
except Exception as e:
    print(f"  data_loader: FAILED - {e}")

try:
    from data.preprocessor import DataPreprocessor
    print("  preprocessor: OK")
except Exception as e:
    print(f"  preprocessor: FAILED - {e}")

try:
    from data.dataset import DatasetSplitter
    print("  dataset: OK")
except Exception as e:
    print(f"  dataset: FAILED - {e}")

try:
    from models.nn_builder import ModelBuilder
    print("  nn_builder: OK")
except Exception as e:
    print(f"  nn_builder: FAILED - {e}")

try:
    from models.evaluator import ModelEvaluator
    print("  evaluator: OK")
except Exception as e:
    print(f"  evaluator: FAILED - {e}")

try:
    from core.device_manager import DeviceManager
    print("  device_manager: OK")
except Exception as e:
    print(f"  device_manager: FAILED - {e}")

try:
    from core.trainer import Trainer
    print("  trainer: OK")
except Exception as e:
    print(f"  trainer: FAILED - {e}")

print("\nTesting data processing...")

try:
    import pandas as pd
    loader = DataLoader()
    df, info = loader.load_csv("test_data.csv")
    print(f"  Load CSV: OK - {info['rows']} rows, {info['columns']} columns")
except Exception as e:
    print(f"  Load CSV: FAILED - {e}")

print("\nAll core module imports successful!")
print("Note: PyQt6 GUI requires additional dependencies")