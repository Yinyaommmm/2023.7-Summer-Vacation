import numpy as np
import DataDealing as dl
import Nerual as nr
import time
import Loss as ls

train_X, train_y, test_X, test_y = dl.createTrainAndTest(size=500)
# dl.drawScatter(train_X, train_y, train_y, description="train")

# set random seed
np.random.seed(42)
epochs = 1600
lr = 0.01
num1 = 20
num2 = 10
nw = nr.Network(loss_func=ls.MAE, batch_size=40, lr=lr, epochs=epochs,)
l1 = nr.FCLayer(in_feature=1, out_feature=num1, act_func=nr.SigmoidFunc)
l2 = nr.FCLayer(in_feature=num1, out_feature=num2, act_func=nr.SigmoidFunc)
l3 = nr.FCLayer(in_feature=num2, out_feature=1, act_func=nr.IdentityFunc)
nw.add(l1)
nw.add(l2)
nw.add(l3)
train_X = np.array(list(map(lambda x: x.reshape(-1, 1), train_X)))
train_y = np.array(list(map(lambda x: x.reshape(-1, 1), train_y)))
test_X = np.array(list(map(lambda x: x.reshape(-1, 1), test_X)))
test_y = np.array(list(map(lambda x: x.reshape(-1, 1), test_y)))


# 训练模型
start_time = time.time()
nw.train(train_X, train_y, test_X, test_y)
end_time = time.time()
total_time = int(end_time - start_time)
print(f'Total time: {total_time}')

# 比较拟合结果
predictY, validY = [], []
for trainData in train_X:
    predictY.append(nw.forward(trainData))
for testData in test_X:
    validY.append(nw.forward(testData))
dl.drawScatter(train_X, train_y, predictY,
               description=f"1-{num1}-{num2}-1 {epochs}train {total_time}s {lr}lr")
dl.drawScatter(test_X, test_y, validY,
               description=f"1-{num1}-{num2}-1 {epochs}train {total_time}s {lr}lr")
