from abc import abstractclassmethod
from abc import ABC
from typing import List
import time
import numpy as np
import NeuralNetwork.ActFunc as act
import NeuralNetwork.Loss as ls
import NeuralNetwork.DataDealing as dl

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
    def specialBP(self, partialLoss: np.ndarray):
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
    # 默认参数满足0,1区间内概率95%
    def __init__(self, in_feature: int, out_feature: int, act_func: act.ActivationFunc, mean=0.5, dev=0.25) -> None:
        # in_feature 输入神经元数量
        # out_feature 输出神经元数量
        # 输入数据
        self.input = np.zeros(shape=(in_feature, 1))
        # 输出结果
        self.output = np.zeros(shape=(out_feature, 1))
        self.funcOutput = np.zeros(shape=(out_feature, 1))  # 经过激活函数的结果
        # 输出求导
        self.partialOutput = np.zeros(shape=(out_feature, 1))
        # 参数
        # self.weight = np.random.randn(in_feature, out_feature)
        # self.bias = np.random.randn(out_feature, 1)
        self.weight = np.random.normal(mean, dev, (in_feature, out_feature))
        self.bias = np.random.normal(mean, dev, (out_feature, 1))
        # 参数梯度
        self.partialWeight = np.zeros(shape=(in_feature, out_feature))
        self.partialBias = np.zeros(shape=(out_feature, 1))
        self.partialFunc = np.zeros(shape=(out_feature, 1))  # 关于激活函数的梯度
        # 激活函数
        self.actFunc = act_func
        # 方差
        self.mean=mean
        self.dev=dev

    def adjustParam(self, lr, batch_size):
        self.weight -= lr * self.partialWeight / batch_size
        self.bias -= lr * self.partialBias / batch_size
        # self.weight -= lr * self.partialWeight
        # self.bias -= lr * self.partialBias

    # 梯度清零，方便下一个Batch
    def clearPartial(self):
        # 只清除累积值
        # self.partialFunc = np.zeros_like(self.partialFunc)
        # self.partialOutput = np.zeros_like(self.partialOutput)
        self.partialWeight = np.zeros_like(self.partialWeight)
        self.partialBias = np.zeros_like(self.partialBias)

    def forward(self, x):
        # 记录输出、输入
        assert (self.input.shape[0] == x.shape[0])
        assert (self.input.shape[1] == x.shape[1])
        self.input = x
        self.output = self.weight.T @ x+self.bias
        self.funcOutput = self.actFunc.forward(self.output)
        return self.funcOutput

    def specialBP(self, partialLoss: np.ndarray):

        # 计算关于输出
        # print('This is SBP')
        # partialLoss  传入损失函数的导数
        # 再计算激活函数导数
        self.partialFunc = self.actFunc.backProp(self.output, self.funcOutput)
        self.partialOutput = partialLoss * self.partialFunc  # 内积

        self.partialWeight += self.input @ self.partialOutput.T  # 使用+=来累积一个batch内的梯度
        self.partialBias += self.partialOutput

    def backProp(self, layers, idx):
        # print("This is normal BP")
        # idx 为当前层原本的层号
        layerNext = layers[idx+1]
        # 激活函数关于输出的求导
        self.partialFunc = self.actFunc.backProp(self.output, self.funcOutput)
        self.partialOutput = (
            layerNext.weight @ layerNext.partialOutput) * self.partialFunc
        self.partialWeight += self.input @ self.partialOutput.T
        self.partialBias += self.partialOutput


# 网络结构


class Network:
    def __init__(self, loss_func: ls.Loss, batch_size=1, lr=0.001, epochs=100,) -> None:
        assert (batch_size >= 1)
        self.layers: List[Layer] = []
        self.loss = 0
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.loss_tendency_x = []
        self.test_loss_tendency_y = []
        self.train_loss_tendency_y = []
        self.train_correct_ratio = []  # 记录正确率
        self.test_correct_ratio = []  # 记录正确率
        self.loss_func = loss_func
        self.amnos = 0
        self.lastEpochos = 0 # 上一次训练的epochos
        self.lastTime = 0

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
        self.loss += self.loss_func.calcLoss(y_hat, y_label)

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
                partialLoss = self.loss_func.partialLoss(
                    layer.funcOutput, labelY)
                layer.specialBP(partialLoss)
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
        start_time = time.time()
        space = int(self.epochs / 20)  # 每5%进行一次测试
        print(f'Start training for {self.epochs} epochs')
        for epoch in range(self.lastEpochos , self.lastEpochos  +self.epochs,1):
            print(f"--------epoch: {epoch}----------")
            self.single_epoch_train(
                trainSet=trainSet, labelSet=trainLabelSet, needBP=True)
            # 每space次验证一次
            if (epoch+1) % space == 0:
                self.loss_tendency_x.append(epoch+1)
                self.validation_loss(testSet,
                                     testLabelSet, epoch, self.test_loss_tendency_y, "Test Loss")
                self.validation_loss(trainSet,
                                     trainLabelSet, epoch, self.train_loss_tendency_y, "Train Loss")
            self.clearLoss()
        end_time = time.time()
        total_time = int(end_time - start_time)
        print(f'This training cost time: {total_time} s')
        print(f'All training cost time: {self.lastTime + total_time} s')
        print(f'Min TotalLoss: {len(trainLabelSet)* min(self.test_loss_tendency_y)} , Min AvgLoss: {min(self.test_loss_tendency_y)}')
        # 更新以往信息
        self.lastEpochos += self.epochs 
        self.lastTime = self.lastTime + total_time 
        # 绘制Loss曲线
        dl.drawPlot(x=self.loss_tendency_x, y1=self.test_loss_tendency_y, y2=self.train_loss_tendency_y,
                    title='Loss Tendency', x_des="Epoch", y_des="Avg Loss", y1_des="Test Loss", y2_des="Train Loss")
        return total_time

    def validation_loss(self, x_set, y_set, epoch, y_containter, vld_des):
        self.single_epoch_train(trainSet=x_set, labelSet=y_set, needBP=False)
        avg_loss = self.loss / x_set.shape[0]
        print(
            f'{vld_des }: {self.loss}  Avg Loss: {avg_loss} Progress: {epoch+1} / {self.epochs}')
        y_containter.append(avg_loss)  # 添加误差

    # 专门用于分类验证
    def classify_validation_loss(self, x_set, y_set, epoch, loss_containter, cr_container, vld_des):
        cr = self.classify_single_epoch_train(
            trainSet=x_set, labelSet=y_set, needBP=False)
        avg_loss = self.loss / x_set.shape[0]
        print(
            f'{vld_des }: {self.loss}  Avg Loss: {avg_loss} Correct Ratio: {cr} Progress: {epoch+1} / {self.lastEpochos + self.epochs}')
        loss_containter.append(avg_loss)  # 添加误差
        cr_container.append(cr)  # 添加正确率

    # 专门用于分类的训练，最后返回计算正确率
    def classify_single_epoch_train(self, trainSet, labelSet, needBP):
        self.clearLoss()
        total_num = len(trainSet)
        correct_num = 0  # 记录正确性
        assert (len(trainSet) == len(labelSet))
        batchCounter = 0
        if needBP:
            for idx, trainData in enumerate(trainSet):
                # 前向传播
                res = self.forward(trainData)
                # 计算误差
                self.calcLoss(res, labelSet[idx])
                # 是否分类正确
                if (np.argmax(res) == np.argmax(labelSet[idx])):
                    correct_num = correct_num + 1
                # 反向传播计算并累积梯度
                self.backProp(labelSet[idx])
                # 一个batch进行梯度更新
                if (idx+1) % self.batch_size == 0:
                    # xx
                    # self.layerInfo(batchCounter)
                    self.batch_AdjustParam(
                        batchCounter, self.loss, self.batch_size, idx, total_num)
                    batchCounter += 1
            # 对于残余部分进行梯度更新
            deltaNum = len(trainSet) - batchCounter*self.batch_size
            if deltaNum > 0:
                self.batch_AdjustParam(
                    batchCounter, self.loss, deltaNum, idx, total_num)
                batchCounter += 1

        else:  # 只进行前向传播 同时计算正确率
            for idx, trainData in enumerate(trainSet):
                # 前向传播
                res = self.forward(trainData)
                # 计算误差
                self.calcLoss(res, labelSet[idx])
                # t_r = np.argmax(res)
                # t_y = np.argmax(labelSet[idx])
                # print(f"idx {idx} res:{t_r} std:{t_y}")
                # 是否分类正确
                if (np.argmax(res) == np.argmax(labelSet[idx])):
                    correct_num = correct_num + 1
        return correct_num / total_num

    def classify_train(self, trainSet, trainLabelSet, testSet, testLabelSet):
        start_time = time.time()
        # space = int(self.epochs / 20)  # 每5%进行一次测试
        space = 5  # 每5epoch进行测试
        # 紧接上一次训练
        print(f'Start training for {self.epochs} epochs')
        for epoch in range(self.lastEpochos,self.lastEpochos + self.epochs , 1):
            print(f"--------epoch: {epoch}----------")
            trainSet, trainLabelSet = dl.data_shuffle(trainSet, trainLabelSet)
            cr = self.classify_single_epoch_train(
                trainSet=trainSet, labelSet=trainLabelSet, needBP=True)
            print(cr)
            # 每space次验证一次
            if (epoch+1) % space == 0:
                self.loss_tendency_x.append(epoch+1)
                self.classify_validation_loss(testSet,
                                              testLabelSet, epoch, self.test_loss_tendency_y, self.test_correct_ratio, "Test Loss")
                self.classify_validation_loss(trainSet,
                                              trainLabelSet, epoch, self.train_loss_tendency_y, self.train_correct_ratio, "Train Loss")
            self.clearLoss()
        end_time = time.time()
        total_time = int(end_time - start_time)
        print(f'This training cost time: {total_time} s')
        print(f'All training cost time: {self.lastTime + total_time} s')
        print(f'Min TotalLoss: {len(trainLabelSet)* min(self.test_loss_tendency_y)} , Min AvgLoss: {min(self.test_loss_tendency_y)}')
        print(f'Max Correct Ratio: {max (self.test_correct_ratio)}, at epoch {(np.argmax(self.test_correct_ratio)+1) * space}')
        self.lastEpochos += self.epochs # 更新lastEpochs
        self.lastTime = self.lastTime + total_time
      
        # 绘制Loss曲线
        dl.drawPlot(x=self.loss_tendency_x, y1=self.test_loss_tendency_y, y2=self.train_loss_tendency_y,
                    title='Loss Tendency', x_des="Epoch", y_des="Avg Loss", y1_des="Test Loss", y2_des="Train Loss")
        # 绘制准确率曲线
        dl.drawPlot(x=self.loss_tendency_x, y1=self.train_correct_ratio, y2=self.test_correct_ratio,
                    title='Correct Ratio Tendency', x_des="Epoch", y_des="Correct Ratio", y1_des="Train CR", y2_des="Test CR")
        return total_time

    def layerInfo(self, batch_counter):
        if self.amnos % 5 == 0:
            print(
                f'No. {self.amnos} adjust from batch {batch_counter}__________________________________)')
            for i in [2, 1, 0]:
                print(f'Layer{i} Partial')
                print(
                    f'{i}-partial weight {self.layers[i].partialWeight[0] / self.batch_size}')
                print(
                    f'{i}-partial bias {self.layers[i].partialBias[0:3]/ self.batch_size}')
                print(f'{i}-partial func {self.layers[i].partialFunc[0:3]}')
                print(f'{i}-PartialOut {self.layers[i].partialOutput[0:10]}\n')
                print(f'{i}-Input {self.layers[i].input[0:10]}\n')
                print(f'{i}-Output {self.layers[i].output[0:12]}\n')
                print(f'{i}-OutputFunc {self.layers[i].funcOutput[0:12]}\n')
                print(f'{i}-Weight {self.layers[i].weight[0]}\n')
                print(f'{i}-Bias {self.layers[i].bias[0:3]}\n')
        self.amnos = self.amnos+1
