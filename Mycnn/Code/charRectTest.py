import NeuralNetwork as nn
import numpy as np

nn.dl.interview_load_test_data('train')

model_name = './提交模型/784-392-196-98-12-m0d0.1-b1-lr0.01'
xx = nn.dl.load_model(model_name)

X_test, y_test = nn.dl.interview_load_test_data(data_dir='train',num_test_samples=240 )
cr =xx.classify_single_epoch_train(X_test, y_test, False)
print(cr)
