import torch.nn as nn


class SCNN(nn.Module):
    def __init__(self,  num_classes, output_feature):
        super(SCNN, self).__init__()
        self.oc = output_feature
        self.conv1 = nn.Sequential(
            # 3* 28 * 28 -> 32* 28 * 28
            nn.Conv2d(3, self.oc, kernel_size=3, padding=1),
            nn.ReLU(),
            # 32 * 28 * 28 -> 32 * 14 * 14
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.oc * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sm(x)
        return x

class DCNN(nn.Module):
    def __init__(self,num_classes, output_feature,k_s =3 ):
        super(DCNN,self).__init__()
        self.oc = output_feature
        self.conv1 = nn.Sequential(
            # 3 * 28 * 28 -> of * 28 * 28  
            nn.Conv2d(3,self.oc,kernel_size=k_s,padding=k_s //2),
            nn.ReLU(),
            # of * 28 * 28 -> of * 14 * 14
            nn.MaxPool2d(kernel_size=2,stride=2)
        ) 
        self.conv2 = nn.Sequential(
            # of * 14 * 14 -> 2of * 14 * 14
            nn.Conv2d(self.oc ,2 * self.oc,kernel_size=k_s,padding=k_s //2),
            nn.ReLU(),
            # 2of * 14 * 14 -> 2of * 7 * 7
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2* self.oc * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.sm(x)
        return x
