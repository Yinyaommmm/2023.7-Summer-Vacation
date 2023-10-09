import numpy as np
import matplotlib.pyplot as plt


def createTrainAndTest(size):
    step = 2*np.pi / size
    X = np.arange(-np.pi, np.pi, step)
    X = np.random.choice(X, size, replace=False)
    train_X = X[0:int(size*0.8)]
    train_X.sort()
    train_y = np.sin(train_X)

    test_X = X[int(size*0.8):]
    test_X.sort()
    # test_X = X
    test_y = np.sin(test_X)

    return train_X, train_y, test_X, test_y


def drawScatter(dataX: np.ndarray, dataY: np.ndarray, predictY: np.ndarray, description: str):
    plt.scatter(dataX, dataY, label='Test Data', color='blue', alpha=0.5, s=5)
    plt.scatter(dataX, predictY, label='Predict Data',
                color='red', alpha=0.5, s=5)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(description)
    plt.legend()
    plt.grid(True)
    plt.show()


def drawPlot(x, y1, y2, title, x_des, y_des, y1_des, y2_des):
    plt.plot(x, y1, label=y1_des, color='blue', marker='o',
             linestyle='-')
    plt.plot(x, y2, label=y2_des, color='red', marker='o',
             linestyle='--')
    # 添加标题和标签
    plt.title(title)  # 图表标题
    plt.xlabel(x_des)  # X轴标签
    plt.ylabel(y_des)  # Y轴标签
    # 显示图表
    plt.legend()
    plt.grid(True)  # 添加网格线
    plt.show()
