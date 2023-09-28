import numpy as np
from abc import abstractclassmethod
from abc import ABC


def ReLu(x):
    return max(0, x)


ReLu_Vectorized = np.vectorize(ReLu)


# 层虚基类
class Layer(ABC):
    @abstractclassmethod
    def forward(self, x):
        # 前向传播
        pass

    def __call__(self, x):
        return self.forward(x)
    
    def getDescription(self):
        #利用str的format格式化字符串
        #利用生成器推导式去获取key和self中key对应的值的集合
        return ",".join("{}={}".format(key,getattr(self,key)) for key in self.__dict__.keys())
    #重写__str__定义对象的打印内容
    def __str__(self):
        return "{}->({})".format(self.__class__.__name__,self.getDescription())


class ReLuLayer(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return ReLu_Vectorized(x)

# 全连接层


class FCLayer(Layer):
    def __init__(self, in_feature, out_feature) -> None:
        # in_feature 输入神经元数量
        # out_feature 输出神经元数量
        self.weight = np.random.randn(in_feature, out_feature)
        self.bias = np.random.randn(out_feature)
        # 输入数据
        self.input = np.zeros(in_feature)
        # 输出结果
        self.output = np.zeros(out_feature)
        # 输出求导
        self.partialOutput = np.zeros(out_feature)
        # 参数梯度
        self.partialWeight = np.zeros(shape=(in_feature,out_feature))
        self.partialBias = np.zeros(out_feature)

    def forward(self, x):
        # 记录输出、输入
        self.input = x
        self.output = self.weight.T @ x+self.bias
        return self.output
    
    def backProp(self):
        self.partialWeight = self.input @ self.partialOutput.T
        self.partialBias =  self.partialOutput.T

# 网络结构
class Network:
    def __init__(self) -> None:
        self.layers = []
    def add(self,layer:Layer):
        self.layers.append(layer)
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x