
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def createTrainAndTest(size):
    step = 2*np.pi / size
    train_size = int(0.8 * size)
    test_size = size - train_size
    X = np.arange(-np.pi, np.pi, step)
    X = np.random.choice(X, size, replace=False)
    train_X = X[0:train_size].reshape(train_size, -1, 1)
    train_y = np.sin(train_X)

    test_X = X[train_size:].reshape(test_size, -1, 1)
    test_y = np.sin(test_X)

    return train_X, train_y, test_X, test_y

# 载入图片
def load_data(data_dir, num_train_samples=500, num_test_samples=120):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for class_label in range(1, 13):
        class_folder = os.path.join(data_dir, str(class_label))

        # images: 存放image名字的list
        images = sorted(os.listdir(class_folder))  # 确保按照文件名的顺序加载图片 
        # 从每个文件夹中加载500个训练样本
        for i in range(num_train_samples):
            image_file = images[i]
            image_path = os.path.join(class_folder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            X_train.append(image)
            y_train.append(class_label - 1)

        # 从每个文件夹中加载120个测试样本
        for i in range(num_train_samples, num_train_samples + num_test_samples):
            image_file = images[i]
            image_path = os.path.join(class_folder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            X_test.append(image)
            y_test.append(class_label - 1)


    # 修改shape 并且转换数据类型
    pic_size = 28
    char_class_num = 12
    X_train = np.array(X_train).reshape(num_train_samples*char_class_num , pic_size**2,1)
    X_test = np.array(X_test).reshape(num_test_samples*char_class_num , pic_size**2,1)
    y_train = np.array(y_train)
    #.reshape(num_train_samples*char_class_num,char_class_num,1)
    y_test = np.array(y_test)

   # 将图像数据归一化到 [0, 1] 的范围
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # # 使用NumPy手动创建one-hot编码
    y_train = np.eye(char_class_num)[y_train].reshape(num_train_samples*char_class_num,char_class_num,1)
    y_test = np.eye(char_class_num)[y_test].reshape(num_test_samples*char_class_num,char_class_num,1)

    return X_train, y_train, X_test, y_test


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
