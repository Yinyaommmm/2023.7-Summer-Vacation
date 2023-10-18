import time
from functools import wraps
import CNNmodel as md
import Draw as dl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch.optim as optim

# 函数计时的注解
def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, pic_per_class=620, needTrain=False, total=620):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))  # 排序后的文件夹名称
        self.pic_per_class = pic_per_class
        self.class_num = 12
        self.needTrain = needTrain
        self.total = total

    def __len__(self):
        # 返回数据集的总长度

        return self.class_num * self.pic_per_class

    def __getitem__(self, idx):
        # 根据索引加载图像和标签
        class_folder = self.classes[idx // self.pic_per_class]
        if self.needTrain:
            img_idx = idx % self.pic_per_class  # 训练集从头开始
        else:
            img_idx = idx % self.pic_per_class + \
                (self.total - self.pic_per_class)  # 验证集跳过训练集

        img_path = os.path.join(
            self.data_dir, class_folder, f"{img_idx + 1}.bmp")
        image = Image.open(img_path).convert("RGB")

        label = int(class_folder)-1   # 注意需要-1

        if self.transform:
            image = self.transform(image)

        return image, label

# 返回训练集和测试集
def createData(batch_size):
    # 数据加载
    data_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    train_set = CustomDataset(
        data_dir='train', transform=data_transform, pic_per_class=500, needTrain=True)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4)

    test_set = CustomDataset(
        data_dir='train', transform=data_transform, pic_per_class=120, needTrain=False)
    test_loader = DataLoader(test_set, batch_size=32,
                             shuffle=True, num_workers=4)

    return train_loader, test_loader

# 训练模型，有BP
def train(model: nn.Module, optimizer: optim.Optimizer, loss_fn, data_loader: DataLoader,log = True):
    model.train()
    correct_num = 0
    total = 0
    loss_accumulate = []
    for batch_images, batch_labels in data_loader:
        total += len(batch_images)
        output = model(batch_images)
        _, predict = torch.max(output, 1)
        correct_num += (predict == batch_labels).sum().item()
        loss = loss_fn(output, batch_labels)
        loss_accumulate.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cr = 100 * correct_num / total
    loss_accumulate = torch.tensor(loss_accumulate).mean()
    if log :
        print(f'Train Loss: {loss_accumulate.item()} , Train CR: {cr:.2f} %')
    return cr, loss.item()

# 测试模型，无BP
def validate(model: nn.Module, loss_fn, data_loader: DataLoader ,log = True):
    model.eval()
    correct_num = 0
    total = 0
    loss_accumulate = []
    with torch.inference_mode():
        for batch_images, batch_labels in data_loader:
            total += len(batch_images)
            output = model(batch_images)
            _, predict = torch.max(output, 1)
            correct_num += (predict == batch_labels).sum().item()
            loss = loss_fn(output, batch_labels)
            loss_accumulate.append(loss.item())

    loss_accumulate = torch.tensor(loss_accumulate).mean()
    cr = 100 * correct_num / total
    if log:
        print(f'Test Loss: {loss_accumulate.item()}, Test CR: {cr:.2f}%')
    return cr, loss.item()

# 训练+测试
@timethis
def MyTrain(load = False):

    # 类别数量设置
    num_classes = 12
    # 超参
    of = 8
    batch_size = 32
    lr = 0.1
    epochs = 60
    kernal_size = 5

    # 加载数据
    train_loader, test_loader = createData(batch_size)

    # 创建模型实例
    model = md.DCNN(num_classes, of,kernal_size)
    model_name = f'D_{of}F_K{kernal_size}_CNN.pth'
    save_dir = f'Double/{of}F/K{kernal_size}'
    save_path = os.path.join(os.getcwd(), 'Part3/DataRecord')
    save_path = os.path.join(save_path, save_dir)
    model_save_path = os.path.join(save_path, model_name)
    if load == True:
        model.load_state_dict(torch.load(model_save_path))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_cr_list = []
    test_cr_list = []
    train_loss_list = []
    test_loss_list = []
    epoch_list = range(1, epochs+1)

    # 训练和测试
    for epoch in range(epochs):
        print(f"------Epoch: {epoch} ------")
        train_cr, train_loss = train(model, optimizer, loss_fn, train_loader,False)
        test_cr, test_loss = validate(model, loss_fn, test_loader,False)
        train_cr_list.append(train_cr)
        test_cr_list.append(test_cr)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    # 训练数据打印
    print(f"-@@@@@@k_s {kernal_size}, of {of}@@@@@-")
    print(f"Min Loss: {torch.min(torch.tensor(test_loss_list))}, at epoch {torch.argmin(torch.tensor(test_loss_list))+1}")
    print(f"Max Acc: {torch.max(torch.tensor(test_cr_list))}, at epoch {torch.argmax(torch.tensor(test_cr_list))+1}")
    # 保存Loss和Acc图以及模型
    print('Save Loss & Acc picture...')
    dl.drawPlot(epoch_list, train_loss_list, test_loss_list,
                "Loss Tendency", 'epoch', 'loss', 'Train Loss', 'Test Loss', True, save_path, 'Loss')
    dl.drawPlot(epoch_list, train_cr_list, test_cr_list, "Correct Ratio",
                'epoch', 'loss', 'Train ACC', 'Test ACC', True, save_path, 'Acc')
    print(f'Save the Model to {model_save_path}')
    torch.save(obj=model.state_dict(),
               f=f'./Part3/DataRecord/{save_dir}/{model_name}')

# 自动化训练+测试
@timethis
def AnaTrain(load = False, oft = 8 , k_s = 3):

    # 类别数量设置
    num_classes = 12
    # 超参
    of = oft
    batch_size = 32
    lr = 0.1
    epochs = 30
    kernal_size = k_s
    print(f"-@@@@@@k_s {kernal_size}, of {of}@@@@@-")
    # 加载数据
    train_loader, test_loader = createData(batch_size)

    # 创建模型实例
    model = md.DCNN(num_classes, of,kernal_size)
    model_name = f'D_{of}F_K{kernal_size}_CNN.pth'
    save_dir = f'Double/{of}F/K{kernal_size}'
    save_path = os.path.join(os.getcwd(), 'Part3/DataRecord')
    save_path = os.path.join(save_path, save_dir)
    model_save_path = os.path.join(save_path, model_name)
    if load == True:
        model.load_state_dict(torch.load(model_save_path))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_cr_list = []
    test_cr_list = []
    train_loss_list = []
    test_loss_list = []
    epoch_list = range(1, epochs+1)

    # 训练和测试
    for epoch in range(epochs):
        print(f"------Epoch: {epoch} ------")
        train_cr, train_loss = train(model, optimizer, loss_fn, train_loader,True)
        test_cr, test_loss = validate(model, loss_fn, test_loader,True)
        train_cr_list.append(train_cr)
        test_cr_list.append(test_cr)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    # 训练数据打印
    print(f"-@@@@@@k_s {kernal_size}, of {of}@@@@@-")
    print(f"Min Loss: {torch.min(torch.tensor(test_loss_list))}, at epoch {torch.argmin(torch.tensor(test_loss_list))+1}")
    print(f"Max Acc: {torch.max(torch.tensor(test_cr_list))}, at epoch {torch.argmax(torch.tensor(test_cr_list))+1}")
    # 保存Loss和Acc图以及模型
    print('Save Loss & Acc picture...')
    dl.drawPlot(epoch_list, train_loss_list, test_loss_list,
                "Loss Tendency", 'epoch', 'loss', 'Train Loss', 'Test Loss', True, save_path, 'Loss')
    dl.drawPlot(epoch_list, train_cr_list, test_cr_list, "Correct Ratio",
                'epoch', 'loss', 'Train ACC', 'Test ACC', True, save_path, 'Acc')
    print(f'Save the Model to {model_save_path}')
    torch.save(obj=model.state_dict(),
               f=f'./Part3/DataRecord/{save_dir}/{model_name}')


if __name__ == '__main__':
    #  MyTrain(False)
    # 16 -5 最佳 98.26%
     ofList = [16]
     ksList = [5]
     for of in ofList:
         for ks in ksList:
              AnaTrain(load=False,oft=of , k_s=ks)
    
 
