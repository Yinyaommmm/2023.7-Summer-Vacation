import numpy as np
import DataCreate as dc
import Nerual as nr


# train_X, train_y, test_X, test_y = dc.createTrainAndTest()


# set random seed 
np.random.seed(42);

l1 = nr.FCLayer(5, 2)
relu = nr.ReLuLayer()
l2 = nr.FCLayer(2, 1)
nw = nr.Network()
nw.add(l1)
nw.add(relu)
nw.add(l2)
X = np.array([1, 2, 3, 4, 0])
X = nw.forward(X)
print(X)

# print("\nThis is the whole process :")
# X = np.array([1, 2, 3, 4, 0])
# print("Origin: ",X)
# X = l1(X)
# print("After L1: ",X)
# X = relu(X)
# print("After relu: ",relu(X))
# X = l2(X)
# print("After L2: ",X)
# print(l1)
