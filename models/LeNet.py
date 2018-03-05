import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim


## network
class MLPNet(nn.Module):
    def __init__(self,input_size,num_classes):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x

class LeNet(nn.Module):
    def __init__(self,num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(16*80, 250)  #63
        self.fc2 = nn.Linear(250, num_classes)
        self.ceriation = nn.CrossEntropyLoss()
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool1d(x, kernel_size=2, stride=4)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        # print x.size()
        x = x.view(-1, 16*80)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    import torch
    from torchvision import models
    model = LeNet(num_classes=5)
    # model = MLPNet(input_size=1024,num_classes=5)
    print model

    num_params = 0
    for parameter in model.parameters():
        temp = 1
        for item in parameter.size():
            temp *= item
        num_params += temp
    print('model: fuknet18')
    print('num_param:', num_params)
    print('==' * 20)


    x = torch.FloatTensor(10,1,2600)
    x = torch.autograd.Variable(x)
    y = model(x)
    print y.size()