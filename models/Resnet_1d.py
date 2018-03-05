#coding=utf8
import torch
from torch import nn
import math
from torchvision.models import resnet



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes,kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes,kernel_size=3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class resnet1d(nn.Module):

    def __init__(self, block, layers, num_classes=1000, in_channels=1):
        self.inplanes = 8
        super(resnet1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=4)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        # self.layer5 = self._make_layer(block, 256, layers[3], stride=2)
        # self.avgpool = nn.AvgPool1d(32)
        self.avgpool = nn.AdaptiveAvgPool1d((1,))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)   #(N,64,533)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.layer5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

class resnet1d_2(nn.Module):

    def __init__(self, block, layers, num_classes=1000, in_channels=1):
        self.inplanes = 2
        super(resnet1d_2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 2, kernel_size=5, stride=4, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(2)
        self.relu1 = nn.ReLU(inplace=True)


        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 4, layers[0])
        self.layer2 = self._make_layer(block, 8, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 16, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 32, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 64, layers[3], stride=2)
        # self.avgpool = nn.AvgPool1d(32)
        self.avgpool = nn.AdaptiveAvgPool1d((1,))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # print x.size()
        x = self.maxpool(x)
        # print x.size()
        x = self.layer1(x)   #(N,64,533)
        # print x.size()
        x = self.layer2(x)
        # print x.size()
        x = self.layer3(x)
        # print x.size()
        x = self.layer4(x)
        # print x.size()
        x = self.layer5(x)
        # print x.size()
        x = self.avgpool(x)
        # print x.size()
        x = x.view(x.size(0), -1)
        # print x.size()

        x = self.fc(x)
        return x

class resnet1d_3(nn.Module):

    def __init__(self, block, layers, num_classes=1000, in_channels=1):
        self.inplanes = 2
        super(resnet1d_3, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 2, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(2)
        self.relu1 = nn.ReLU(inplace=True)


        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 4, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 8, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 16, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 32, layers[3], stride=2)
        # self.layer5 = self._make_layer(block, 64, layers[3], stride=2)
        # self.avgpool = nn.AvgPool1d(32)
        self.avgpool = nn.AdaptiveAvgPool1d((1,))
        self.fc = nn.Linear(32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # print x.size()
        x = self.maxpool(x)
        # print x.size()
        x = self.layer1(x)   #(N,64,533)
        # print x.size()
        x = self.layer2(x)
        # print x.size()
        x = self.layer3(x)
        # print x.size()
        x = self.layer4(x)
        # print x.size()
        # x = self.layer5(x)
        # print x.size()
        x = self.avgpool(x)
        # print x.size()
        x = x.view(x.size(0), -1)
        # print x.size()

        x = self.fc(x)
        return x




def resnet20_1d( **kwargs):
    model = resnet1d(BasicBlock, [2, 2, 2, 2, 2], **kwargs)
    return model

def resnet20_1d2( **kwargs):
    model = resnet1d_2(BasicBlock, [2, 2, 2, 2, 2], **kwargs)
    return model

def resnet20_1d3( **kwargs):
    model = resnet1d_2(BasicBlock, [1, 1, 1, 1, 1], **kwargs)
    return model

def resnet20_1d4( **kwargs):
    model = resnet1d_3(BasicBlock, [1, 1, 1, 1, 1], **kwargs)
    return model

def resnet34_1d(**kwargs):
    model = resnet1d(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet34_1d2(**kwargs):
    model = resnet1d_2(BasicBlock, [3, 4, 6,4, 3], **kwargs)
    return model

def resnet50_1d( **kwargs):
    model = resnet1d(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

if __name__ == '__main__':
    # from torchvision import models
    # model = BasicBlock(inplanes=64, planes=64)
    # model = fuknet(BasicBlock, [2, 2, 2, 2], 2)
    # print model

    model = resnet20_1d(num_classes=4)
    print model

    x = torch.FloatTensor(10,1,2600)
    x = torch.autograd.Variable(x)

    y = model(x)

    para_cnt = 0
    for item in model.state_dict().keys():
        para_size = model.state_dict()[item].size()
        cnt = 1
        for s in para_size:
            cnt = cnt * s

        para_cnt += cnt

    print para_cnt

    # print y.size()