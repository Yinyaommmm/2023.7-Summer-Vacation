import numpy as np
import NerualNetwork as nn

train_X, train_y, test_X, test_y = nn.dl.createTrainAndTest(size=500)
# dl.drawScatter(train_X, train_y, train_y, description="train")

# set random seed
np.random.seed(42)
epochs = 2400
lr = 0.01
num1 = 20
num2 = 10
batch_size = 1
nw = nn.Network(loss_func=nn.ls.MAE, batch_size=batch_size,
                lr=lr, epochs=epochs,)
l1 = nn.FCLayer(in_feature=1, out_feature=num1, act_func=nn.act.Sigmoid)
l2 = nn.FCLayer(in_feature=num1, out_feature=num2, act_func=nn.act.Sigmoid)
l3 = nn.FCLayer(in_feature=num2, out_feature=1, act_func=nn.act.Idempotent)
nw.add(l1)
nw.add(l2)
nw.add(l3)
# 训练模型
total_time = 0
total_time = nw.train(train_X, train_y, test_X, test_y)


# 比较拟合结果
predictY, validY = [], []
for trainData in train_X:
    predictY.append(nw.forward(trainData))
for testData in test_X:
    validY.append(nw.forward(testData))
nn.dl.drawScatter(train_X, train_y, predictY,
                  description=f"1-{num1}-{num2}-1 {epochs}train {total_time}s {lr}lr")
nn.dl.drawScatter(test_X, test_y, validY,
                  description=f"1-{num1}-{num2}-1 {epochs}train {total_time}s {lr}lr")
