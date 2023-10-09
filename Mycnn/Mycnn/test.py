import Nerual as nr
import numpy as np
import matplotlib.pyplot as plt
import DataDealing as dl
import Loss as ls
x = np.array([1, 5]).reshape(-1, 1)
y = np.array([2, 0]).reshape(-1, 1)
t = ls.MAE(x, y)
print(t)
print(t.sum())

ssss