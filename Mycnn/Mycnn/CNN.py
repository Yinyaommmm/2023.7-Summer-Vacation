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

def train(model:nn.Module , optimizer:optim.Optimizer, loss_fn, data_loader:DataLoader):
    model.train()
    correct_num = 0
    for batch_images, batch_labels in data_loader:
        output = model(batch_images)
        _,predict  = torch.max(output,1)
        correct_num += (predict == batch_labels).sum().item()
        loss = loss_fn(output,batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    cr = 100* correct_num / 12 / 500
    print(f'Train Loss: {loss.item()} , Train CR: {cr:.2f} %')
    return cr,loss.item()

def validate(model:nn.Module ,loss_fn , data_loader:DataLoader):
    model.eval()
    correct_num = 0
    with torch.inference_mode():
        for batch_images, batch_labels in data_loader:
            output = model(batch_images)
            _,predict  = torch.max(output,1)
            correct_num += (predict == batch_labels).sum().item()
            loss = loss_fn(output,batch_labels)

    cr = 100* correct_num / 12 / 120
    print(f'Test Loss: {loss.item()}, Test CR: {cr:.2f}%')
    return cr,loss.item()
    
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
        train(model,optimizer,loss_fn,train_loader)
        validate(model,loss_fn,test_loader)
      

if __name__ == '__main__':
    mytrain()
