import torch.nn as nn


class SCNN(nn.Module):
    def __init__(self,  num_classes, output_feature,):
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
