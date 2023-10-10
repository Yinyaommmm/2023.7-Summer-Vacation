import numpy as np
from abc import abstractclassmethod
from abc import ABC


class ActivationFunc(ABC):
    @classmethod
    def forward(self, funcInput: np.ndarray):
        pass

    @classmethod
    def backProp(self, funcInput: np.ndarray, funcOutput: np.ndarray):
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
    def forward(self, funcInput: np.ndarray):
        # 返回正向传播结果
        return ReLu_Vectorized(funcInput)

    @classmethod
    def backProp(self, funcInput: np.ndarray, funcOutput: np.ndarray):
        return ReLuPartial_Vectorized(funcInput)
# 幂等映射


class Idempotent(ActivationFunc):
    @classmethod
    def forward(self, funcInput: np.ndarray):
        assert (funcInput.shape[1] == 1)  # 确保是个列向量
        return funcInput

    @classmethod
    def backProp(self, funcInput: np.ndarray, funcOutput: np.ndarray):
        return np.ones_like(funcInput)

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
    def forward(self, funcInput: np.ndarray):
        # 返回正向传播结果
        return Sigmoid_Vectorized(funcInput)

    @classmethod
    def backProp(self, funcInput: np.ndarray, funcOutput: np.ndarray):
        return funcOutput*(1-funcOutput)


class Softmax(ActivationFunc):
    @classmethod
    def forward(self, funcInput: np.ndarray):
        e_funcInput = np.exp(funcInput - np.max(funcInput))
        return e_funcInput / np.sum(e_funcInput)

    @classmethod
    def backProp(self, funcInput: np.ndarray, funcOutput: np.ndarray):
        # Softmax的导数放在CE里计算了，这里直接掠过
        return np.ones_like(funcInput)
