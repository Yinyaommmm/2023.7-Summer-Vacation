import numpy as np
import DataDealing as dc
import Nerual as nr


# train_X, train_y, test_X, test_y = dc.createTrainAndTest()


# set random seed
np.random.seed(42)

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([2, 3, 4, 5, 6])
trainSet = np.array([x1, x2])
# 转换成列向量的数组
trainSet = np.array(list(map(lambda x: x.reshape(-1, 1), trainSet)))

y1 = np.array([-9])
y2 = np.array([2])
labelSet = np.array([y1, y2])
labelSet = np.array(list(map(lambda x: x.reshape(-1, 1), labelSet)))


l1 = nr.FCLayer(in_feature=5, out_feature=2, act_func=nr.ReLuFunc)
l3 = nr.FCLayer(in_feature=2, out_feature=2, act_func=nr.ReLuFunc)
l2 = nr.FCLayer(in_feature=2, out_feature=1, act_func=nr.IdentityFunc)
nw = nr.Network(batch_size=1, lr=0.001, epochs=5001)
nw.add(l1)
# nw.add(l3)
nw.add(l2)
nw.train(trainSet=trainSet, labelSet=labelSet)
