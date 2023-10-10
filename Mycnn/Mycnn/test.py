import NerualNetwork as nn
import numpy as np

x = np.array([0.1, 0.9, 0.5]).reshape(-1, 1)
# y = np.array([0, 1, 0]).reshape(-1, 1)
# print(nn.ls.CE.partialLoss(x, x, y))

print(nn.act.Sigmoid.backProp(x, x))
