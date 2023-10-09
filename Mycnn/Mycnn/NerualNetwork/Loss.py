import numpy as np
from abc import abstractclassmethod
from abc import ABC


class Loss(ABC):
    @abstractclassmethod
    def calcLoss(y_hat: np.ndarray, y_label: np.ndarray):
        pass


def MAE(y_hat: np.ndarray, y_label: np.ndarray):
    return np.abs(y_hat-y_label).sum()