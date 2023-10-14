import NerualNetwork as nn
import numpy as np
import pickle

np.random.seed(42)
pic_size = 28 * 28
char_class_num = 12
epochs = 600
lr = 0.001
num1 = 128
num2 = 64
batch_size = 100
mean = 0
dev = 0.1

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

print(nw.layers[0].weight[0][:10])


def save_param(nw: nn.Network, filename):

    params = {
        "network": nw
    }

    with open(filename, 'wb') as file:
        pickle.dump(params, file)


def load_model(filename):
    with open(filename, 'rb') as file:
        model_params = pickle.load(file)
    return model_params['network']


save_param(nw, 'aloha.mp')
print('finish')
xx = load_model('aloha.mp')
