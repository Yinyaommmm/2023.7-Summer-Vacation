import NerualNetwork as nn
import numpy as np


# 将所有图片载入数据
X_train, y_train, X_test, y_test = nn.dl.load_data(
    "train", num_train_samples=50, num_test_samples=2)
# X_train (6000,784,1) X_test (1440,784,1)
# y_train (6000,12,1) y_test (1440,784,1)


np.random.seed(42)
pic_size = 28 * 28
char_class_num = 12
epochs = 100
lr = 0.001
num1 = 128
num2 = 32
batch_size = 10
nw = nn.Network(loss_func=nn.ls.CE, batch_size=batch_size,
                lr=lr, epochs=epochs,)
l1 = nn.FCLayer(in_feature=pic_size, out_feature=num1,
                act_func=nn.act.ReLu, mean=0.005, dev=0.0025)
l2 = nn.FCLayer(in_feature=num1, out_feature=num2,
                act_func=nn.act.ReLu, mean=0.005, dev=0.0025)
l3 = nn.FCLayer(in_feature=num2, out_feature=char_class_num,
                act_func=nn.act.Softmax, mean=0.005, dev=0.0025)
nw.add(l1)
nw.add(l2)
nw.add(l3)

total_time = 0

total_time = nw.classify_train(X_train, y_train, X_test, y_test)
print("layer 2")
print(nw.layers[2].input)
print(nw.layers[2].output)
print(nw.layers[2].bias)
print(np.argmax((nw.layers[2].funcOutput)))
print("layer 1")
print(nw.layers[1].input)
print(nw.layers[1].output)
print(nw.layers[1].bias)
