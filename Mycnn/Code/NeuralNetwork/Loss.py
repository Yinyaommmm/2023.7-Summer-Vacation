import numpy as np
from abc import abstractclassmethod
from abc import ABC


class Loss(ABC):
    @abstractclassmethod
    def calcLoss(y_hat: np.ndarray, y_label: np.ndarray):
        pass

    @abstractclassmethod
    def partialLoss(funcOutput: np.ndarray, y_label: np.ndarray):
        pass


class MAE(Loss):
    def calcLoss(y_hat: np.ndarray, y_label: np.ndarray):
        return np.abs(y_hat - y_label).sum()

    # 采用MSE的梯度
    def partialLoss(funcOutput: np.ndarray, y_label: np.ndarray):
        return funcOutput - y_label


class MSE(Loss):
    def calcLoss(y_hat: np.ndarray, y_label: np.ndarray):
        return 0.5 * np.sum((y_hat - y_label)**2)

    def partialLoss(funcOutput: np.ndarray, y_label: np.ndarray):
        return funcOutput - y_label


class CE(Loss):
    def calcLoss(y_hat: np.ndarray, y_label: np.ndarray):
        delta = 1e-7  # 防止出现log(0)
        return -np.sum(y_label*np.log(y_hat + delta))

    def partialLoss(funcOutput: np.ndarray, y_label: np.ndarray):
        # CE
        return funcOutput - y_label
