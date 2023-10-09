import numpy as np
from abc import abstractclassmethod
from abc import ABC
from typing import List
import DataDealing as dl
# 激活函数基类


class ActivationFunc:
    @classmethod
    def forward(self, x):
        pass

    @classmethod
    def backProp(self, x):
        pass

# ReLU部分


def ReLu(x):
    return max(0, x)


def ReLuPartial(x):
    return 1 if x >= 0 else 0


ReLu_Vectorized = np.vectorize(ReLu)
ReLuPartial_Vectorized = np.vectorize(ReLuPartial)


class ReLuFunc(ActivationFunc):
    @classmethod
    def forward(self, x: np.ndarray):
        # 返回正向传播结果
        return ReLu_Vectorized(x)

    @classmethod
    def backProp(self, x: np.ndarray):
        return ReLuPartial_Vectorized(x)
# 幂等映射


class IdentityFunc(ActivationFunc):
    @classmethod
    def forward(self, x: np.ndarray):
        assert (x.shape[1] == 1)  # 确保是个列向量
        return x

    @classmethod
    def backProp(self, x: np.ndarray):
        return np.ones_like(x)

# Sigmoid


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def SigmoidPartial(x):
    y = Sigmoid(x)
    return y*(1-y)


Sigmoid_Vectorized = np.vectorize(Sigmoid)
SigmoidPartial_Vectorized = np.vectorize(SigmoidPartial)


class SigmoidFunc(ActivationFunc):
    @classmethod
    def forward(self, x: np.ndarray):
        # 返回正向传播结果
        return Sigmoid_Vectorized(x)

    @classmethod
    def backProp(self, x: np.ndarray):
        return SigmoidPartial_Vectorized(x)
# 层虚基类


class Layer(ABC):
    @abstractclassmethod
    def adjustParam(self, lr):
        pass

    @abstractclassmethod
    def forward(self, x):
        # 前向传播
        pass

    @abstractclassmethod
    def specialBP(self, x):
        pass

    @abstractclassmethod
    def backProp(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)

    def getDescription(self):
        # 利用str的format格式化字符串
        # 利用生成器推导式去获取key和self中key对应的值的集合
        return ",".join("{}={}".format(key, getattr(self, key)) for key in self.__dict__.keys())

    # 重写__str__定义对象的打印内容

    def __str__(self):
        return "{}->({})".format(self.__class__.__name__, self.getDescription())


# 全连接层


class FCLayer(Layer):
    def __init__(self, in_feature: int, out_feature: int, act_func: ActivationFunc) -> None:
        # in_feature 输入神经元数量
        # out_feature 输出神经元数量
        # 输入数据
        self.input = np.zeros(shape=(in_feature, 1))
        # 输出结果
        self.output = np.zeros(shape=(out_feature, 1))
        # 输出求导
        self.partialOutput = np.zeros(shape=(out_feature, 1))
        # 参数
        self.weight = np.random.randn(in_feature, out_feature)
        self.bias = np.random.randn(out_feature, 1)
        # 参数梯度
        self.partialWeight = np.zeros(shape=(in_feature, out_feature))
        self.partialBias = np.zeros(shape=(out_feature, 1))
        self.partialFunc = np.zeros(shape=(out_feature, 1))  # 关于激活函数的梯度
        # 激活函数
        self.actFunc = act_func

    def adjustParam(self, lr, batch_size):
        self.weight -= lr * self.partialWeight / batch_size
        self.bias -= lr * self.partialBias / batch_size

    # 梯度清零，方便下一个Batch
    def clearPartial(self):
        self.partialBias = np.zeros_like(self.partialBias)
        self.partialFunc = np.zeros_like(self.partialFunc)
        self.partialOutput = np.zeros_like(self.partialOutput)
        self.partialWeight = np.zeros_like(self.partialWeight)

    def forward(self, x):
        # 记录输出、输入

        assert (self.input.shape[0] == x.shape[0])
        assert (self.input.shape[1] == x.shape[1])
        self.input = x
        self.output = self.weight.T @ x+self.bias
        return self.actFunc.forward(self.output)

    def specialBP(self, labelY):

        # 计算关于输出
        # print('This is SBP')
        partialLoss = (self.output - labelY)  # MSE的导数
        # 计算激活函数导数
        self.partialFunc = self.actFunc.backProp(self.output)
        self.partialOutput = partialLoss * self.partialFunc  # 内积

        self.partialWeight += self.input @ self.partialOutput.T  # 使用+=来累积一个batch内的梯度
        self.partialBias += self.partialOutput

    def backProp(self, layers, idx):
        # print("This is normal BP")
        # idx 为当前层原本的层号
        layerNext = layers[idx+1]
        # 激活函数关于输出的求导
        self.partialFunc = self.actFunc.backProp(self.output)
        self.partialOutput = (
            layerNext.weight @ layerNext.partialOutput) * self.partialFunc
        self.partialWeight += self.input @ self.partialOutput.T
        self.partialBias += self.partialOutput


# 网络结构


class Network:
    def __init__(self, loss_func, batch_size=1, lr=0.001, epochs=100,) -> None:
        assert (batch_size >= 1)
        self.layers: List[Layer] = []
        self.loss = 0
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.loss_tendency_x = []
        self.test_loss_tendency_y = []
        self.train_loss_tendency_y = []
        self.loss_func = loss_func

    def add(self, layer: Layer):
        self.layers.append(layer)

    def adjustParam(self, batch_size):
        for layer in self.layers:
            layer.adjustParam(self.lr, batch_size)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x

    def calcLoss(self, y_hat, y_label):
        self.loss += self.loss_func(y_hat, y_label)

    # 所有层的梯度清零
    def clearPartial(self):
        for layer in self.layers:
            layer.clearPartial()

    def clearLoss(self):
        self.loss = 0

    #  最后一层特殊BP，其...............................余层正常BP
    def backProp(self, labelY):
        layerNum = len(self.layers)
        assert (layerNum >= 1)
        for idx, layer in enumerate(reversed(self.layers)):
            if idx == 0:
                layer.specialBP(labelY)
            else:
                # layerNum-1-idx是原来其在网络结构中的序号，方便利用nextLayer的数据
                layer.backProp(self.layers, layerNum-1-idx)

    def single_epoch_train(self, trainSet, labelSet, needBP):
        self.clearLoss()
        total_num = len(trainSet)
        assert (len(trainSet) == len(labelSet))
        batchCounter = 0
        if needBP:
            for idx, trainData in enumerate(trainSet):
                # 前向传播
                res = self.forward(trainData)
                # 计算误差
                self.calcLoss(res, labelSet[idx])

                # 反向传播计算并累积梯度
                self.backProp(labelSet[idx])
                # 一个batch进行梯度更新
                if (idx+1) % self.batch_size == 0:
                    self.batch_AdjustParam(
                        batchCounter, self.loss, self.batch_size, idx, total_num)
                    batchCounter += 1
            # 对于残余部分进行梯度更新
                deltaNum = len(trainSet) - batchCounter*self.batch_size
                if deltaNum > 0:
                    self.batch_AdjustParam(
                        batchCounter, self.loss, deltaNum, idx, total_num)
                    batchCounter += 1
        else:  # 只进行前向传播
            for idx, trainData in enumerate(trainSet):
                # 前向传播
                res = self.forward(trainData)
                # 计算误差
                self.calcLoss(res, labelSet[idx])

    # 对batch进行参数更新，最后清除
    def batch_AdjustParam(self, batch_Idx, loss, batch_size, idx, total_num):
        # print(
        #     f'Batch {batch_Idx} finished, Loss is {loss}, Progress {idx+1} / {total_num}')
        # 每一层都进行参数调整
        self.adjustParam(batch_size)
        self.clearPartial()

    def train(self, trainSet, trainLabelSet, testSet, testLabelSet):
        space = self.epochs / 20  # 每5%进行一次测试
        for epoch in range(self.epochs+1):
            print(f"--------epoch: {epoch}----------")
            self.single_epoch_train(
                trainSet=trainSet, labelSet=trainLabelSet, needBP=True)
            # # 每space次验证一次
            if (epoch+1) % space == 0:
                self.loss_tendency_x.append(epoch+1)
                self.validation_loss(testSet,
                                     testLabelSet, epoch, self.test_loss_tendency_y, "Test Loss")
                self.validation_loss(trainSet,
                                     trainLabelSet, epoch, self.train_loss_tendency_y, "Train Loss")
            self.clearLoss()
        # 绘制Loss曲线
        dl.drawPlot(x=self.loss_tendency_x, y1=self.test_loss_tendency_y, y2=self.train_loss_tendency_y,
                    title='Loss Tendency', x_des="Epoch", y_des="Avg Loss", y1_des="Test Loss", y2_des="Train Loss")

    def validation_loss(self, x_set, y_set, epoch, y_containter, vld_des):
        self.single_epoch_train(trainSet=x_set, labelSet=y_set, needBP=False)
        avg_loss = self.loss / x_set.shape[0]
        print(
            f'{vld_des } :{self.loss}  Avg Loss :{avg_loss} Progress: {epoch+1} / {self.epochs}')
        y_containter.append(avg_loss)  # 添加误差
