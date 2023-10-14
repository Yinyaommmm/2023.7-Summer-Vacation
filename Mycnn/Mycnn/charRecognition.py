import NeuralNetwork as nn
import numpy as np
import sys


# 将所有图片载入数据
X_train, y_train, X_test, y_test = nn.dl.load_data(
    "train", num_train_samples=500, num_test_samples=120)
# X_train (6000,784,1) X_test (1440,784,1)
# y_train (6000,12,1) y_test (1440,784,1)


# Hyper Parameter
np.random.seed(42)
pic_size = 28 * 28
char_class_num = 12
epochs = 40
lr = 0.01
num1 = 260
num2 = 86
batch_size = 1
mean = 0
dev = 0.2

# Build Network
nw = nn.Network(loss_func=nn.ls.CE, batch_size=batch_size,
                lr=lr, epochs=epochs,)
l1 = nn.FCLayer(in_feature=pic_size, out_feature=num1,
                act_func=nn.act.ReLu, mean=mean, dev=dev)
l2 = nn.FCLayer(in_feature=num1, out_feature=num2,
                act_func=nn.act.ReLu, mean=mean, dev=dev)
l3 = nn.FCLayer(in_feature=num2, out_feature=char_class_num,
                act_func=nn.act.Softmax, mean=mean, dev=dev)
nw.add(l1)
nw.add(l2)
nw.add(l3)

# sys.stdout = open('output.txt', 'w')
total_time = 0
total_time = nw.classify_train(X_train, y_train, X_test, y_test)

nn.dl.save_model(nw,f'{pic_size}-{num1}-{num2}-{char_class_num}-m{mean}d{dev}-b{batch_size}-lr{lr}')


# # 读取过往模型
# model_name = '784-392-196-12-m0d0.2-b1-lr0.01'
# xx = nn.dl.load_model(model_name);
# xx.epochs = 10
# xx.lr = 0.001
# xx.classify_train(X_train, y_train, X_test, y_test)
# nn.dl.save_model(xx,model_name)
