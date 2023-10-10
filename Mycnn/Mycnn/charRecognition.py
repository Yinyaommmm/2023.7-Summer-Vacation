import NerualNetwork as nn
import numpy as np


X_train, y_train_onehot, X_test, y_test_onehot = nn.dl.load_data("train")
print(len(X_train))

# 将所有图片载入数据
