import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None , pic_per_class = 620 , needTrain=False, total = 620):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir)) # 排序后的文件夹名称
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
            img_idx = idx % self.pic_per_class # 训练集从头开始
        else:
            img_idx = idx % self.pic_per_class + (self.total - self.pic_per_class) # 验证集跳过训练集

        img_path = os.path.join(self.data_dir, class_folder, f"{img_idx + 1}.bmp")
        image = Image.open(img_path).convert("RGB")
        
        label = int(class_folder)-1   # 注意需要-1
    
        if self.transform:
            image = self.transform(image)

        return image, label


class CNNModel(nn.Module):
    def __init__(self, num_classes):
            super(CNNModel, self).__init__()
            self.oc = 32
            self.conv1 = nn.Conv2d(3, self.oc, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(self.oc * 14 * 14, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sm(x)
        return x

def createData():
     # 数据加载
    data_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    train_set = CustomDataset(data_dir='train',transform=data_transform,pic_per_class=500,needTrain=True)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    
    test_set = CustomDataset(data_dir='train',transform=data_transform,pic_per_class=120,needTrain=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=4)

    return train_loader,test_loader

def train(model:nn.Module , optimizer:optim.Optimizer, loss_fn:nn._WeightedLoss , data_loader:DataLoader):
    model.train()
    correct_num = 0
    for batch_images, batch_labels in data_loader:
        output = model(batch_images)
        _,predict  = torch.max(output,1)
        correct_num += (predict == batch_labels).sum().item()
        loss = loss_fn(predict,batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    cr = correct_num / 12 / 500
    print(f'Train Loss: {loss.item()} , Train CR: {cr:.2f} %')

def validate(model:nn.Module ,loss_fn:nn._WeightedLoss , data_loader:DataLoader):
    model.eval()
    correct_num = 0
    with torch.inference_mode:
        for batch_images, batch_labels in data_loader:
            output = model(batch_images)
            _,predict  = torch.max(output,1)
            correct_num += (predict == batch_labels).sum().item()
            loss = loss_fn(predict,batch_labels)

    cr = correct_num / 12 / 120
    print(f'Test Loss: {loss.item()}, Test CR: {cr:.2f}%')
    
def mytrain():
    train_loader,test_loader = createData()

    # 创建模型实例
    num_classes = 12  # 类别数量设置
    model = CNNModel(num_classes)
   
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.4) 
    eopchs = 20

    for epoch in range(eopchs):
        print(f"------Epoch: {epoch} ------")
        model.train()
        train_correct_num = 0
        train_loss = torch.tensor(0) 
        for batch_images, batch_labels in train_loader:
     
            train_outputs = model(batch_images)
            _,train_predict = torch.max(train_outputs,dim=1)
            train_correct_num += (train_predict == batch_labels).sum().item()
            train_loss = loss_fn(train_outputs, batch_labels)
            optimizer.zero_grad()  # 清零梯度
            train_loss.backward()
            optimizer.step()


        model.eval()
        test_correct_num = 0
        test_loss = 0
        for batch_images, batch_labels in test_loader:
            test_outputs = model(batch_images)
            test_loss = loss_fn(test_outputs,batch_labels
                            )
            _, test_predicted = torch.max(test_outputs, dim=1)  # 获取预测结果
            # print('output')
            # print(test_outputs)
            # print('predict')
            # print(test_predicted)
            # print('label')
            # print(batch_labels)
            # print('loss')
            # print(test_loss)
            test_correct_num += (test_predicted == batch_labels).sum().item()
           
        # 打印训练和测试数据信息
        Train_CR = 100 * train_correct_num / 12 / 500
        Test_CR = 100 * test_correct_num / 12 / 120

        print(f'Epoch [{epoch + 1}/{eopchs}], Train Loss: {train_loss.item()} , Train CR: {Train_CR:.2f} %')
        print(f'Epoch [{epoch + 1}/{eopchs}], Test Loss: {test_loss.item()} , Test CR: {Test_CR:.2f} %')
      

if __name__ == '__main__':
    mytrain()
    #  train_loader,test_loader = createData()
    #  print(len(test_loader))
