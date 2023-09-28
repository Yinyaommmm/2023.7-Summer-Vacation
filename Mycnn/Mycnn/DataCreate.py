import numpy as np


def createTrainAndTest():

    train_X = np.arange(-np.pi, np.pi, 0.02)
    train_y = np.sin(train_X)

    test_X = np.arange(int(train_X.size*0.2))
    test_X = test_X*np.pi
    test_y = np.sin(test_X)
    return train_X, train_y, test_X, test_y
