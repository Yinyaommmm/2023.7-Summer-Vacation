import numpy as np
from abc import abstractclassmethod
from abc import ABC

class ActivationFunc(ABC):
    @classmethod
    def forward(self, x):
        pass

    @classmethod
    def backProp(self, x):
        pass

# ReLU部分


def ReLU_Single(x):
    return max(0, x)


def ReLuPartial_Single(x):
    return 1 if x >= 0 else 0


ReLu_Vectorized = np.vectorize(ReLU_Single)
ReLuPartial_Vectorized = np.vectorize(ReLuPartial_Single)


class ReLu(ActivationFunc):
    @classmethod
    def forward(self, x: np.ndarray):
        # 返回正向传播结果
        return ReLu_Vectorized(x)

    @classmethod
    def backProp(self, x: np.ndarray):
        return ReLuPartial_Vectorized(x)
# 幂等映射


class Idempotent(ActivationFunc):
    @classmethod
    def forward(self, x: np.ndarray):
        assert (x.shape[1] == 1)  # 确保是个列向量
        return x

    @classmethod
    def backProp(self, x: np.ndarray):
        return np.ones_like(x)

# Sigmoid


def Sigmoid_Single(x):
    return 1 / (1 + np.exp(-x))


def SigmoidPartial_Single(x):
    y = Sigmoid_Single(x)
    return y*(1-y)


Sigmoid_Vectorized = np.vectorize(Sigmoid_Single)
SigmoidPartial_Vectorized = np.vectorize(SigmoidPartial_Single)


class Sigmoid(ActivationFunc):
    @classmethod
    def forward(self, x: np.ndarray):
        # 返回正向传播结果
        return Sigmoid_Vectorized(x)

    @classmethod
    def backProp(self, x: np.ndarray):
        return SigmoidPartial_Vectorized(x)