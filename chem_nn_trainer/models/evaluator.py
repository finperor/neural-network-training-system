import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelEvaluator:
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
                task_type: str = "regression") -> Dict[str, float]:
        if task_type in ("regression", "回归"):
            return ModelEvaluator._regression_metrics(y_true, y_pred)
        else:
            return ModelEvaluator._classification_metrics(y_true, y_pred)

    @staticmethod
    def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape)
        }

    @staticmethod
    def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = y_true.flatten().astype(int)
        y_pred = y_pred.flatten().astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    @staticmethod
    def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.numpy())

        predictions = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        return targets, predictions