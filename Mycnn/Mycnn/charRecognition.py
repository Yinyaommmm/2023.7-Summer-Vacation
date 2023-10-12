import NerualNetwork as nn
import numpy as np
import sys


# 将所有图片载入数据
X_train, y_train, X_test, y_test = nn.dl.load_data(
    "train", num_train_samples=500, num_test_samples=120)
# X_train (6000,784,1) X_test (1440,784,1)
# y_train (6000,12,1) y_test (1440,784,1)


np.random.seed(42)
pic_size = 28 * 28
char_class_num = 12
epochs = 30
lr = 0.01
num1 = 128
num2 = 64
batch_size = 100
nw = nn.Network(loss_func=nn.ls.CE, batch_size=batch_size,
                lr=lr, epochs=epochs,)
l1 = nn.FCLayer(in_feature=pic_size, out_feature=num1,
                act_func=nn.act.ReLu, mean=0.1, dev=0.25)
l2 = nn.FCLayer(in_feature=num1, out_feature=num2,
                act_func=nn.act.ReLu, mean=0.1, dev=0.25)
l3 = nn.FCLayer(in_feature=num2, out_feature=char_class_num,
                act_func=nn.act.Softmax, mean=0.1, dev=0.25)
nw.add(l1)
nw.add(l2)
nw.add(l3)

# sys.stdout = open('output.txt', 'w')
total_time = 0
total_time = nw.classify_train(X_train, y_train, X_test, y_test)

# c = 0
# for idx, testData in enumerate(X_test):
#     res = nw.forward(testData)
#     t_r = np.argmax(res)
#     t_y = np.argmax(y_test[idx])
#     print(f"idx {idx} res:{t_r} std:{t_y}")
#     if (t_r == t_y):
#         c = c+1
# print(f"Cr: {c/len(X_test)}")
