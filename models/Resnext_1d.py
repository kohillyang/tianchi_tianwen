import torch.nn as nn
import math
import torch

# __all__ = ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101',
#            'resnext152']

def conv3x3(in_planes, out_planes, stride=1,groups=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False,groups=groups)


class BasicBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride=stride)
        self.bn1 = nn.BatchNorm1d(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.bn2 = nn.BatchNorm1d(planes*2)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes*2)
        self.conv2 = nn.Conv1d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm1d(planes*2)
        self.conv3 = nn.Conv1d(planes*2, planes * 4, kernel_size=1, bias=False)
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

class ResNeXt_type1(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_group=8):
        self.inplanes = 16
        super(ResNeXt_type1, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], num_group, stride=2)
        self.layer2 = self._make_layer(block, 32, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], num_group, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=(1,))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride=stride, downsample=downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNeXt_type2(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_group=3):
        self.inplanes = 6
        super(ResNeXt_type2, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(6)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 12, layers[0], num_group)
        self.layer2 = self._make_layer(block, 24, layers[1], num_group, stride=4)
        self.layer3 = self._make_layer(block, 48, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 96, layers[3], num_group, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=(1,))
        self.fc = nn.Linear(96 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride=stride, downsample=downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class ResNeXt_type3(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_group=4):
        self.inplanes = 8
        super(ResNeXt_type3, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], num_group)
        self.layer2 = self._make_layer(block, 32, layers[1], num_group, stride=4)
        self.layer3 = self._make_layer(block, 64, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], num_group, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=(1,))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride=stride, downsample=downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnext20_type1( **kwargs):
    model = ResNeXt_type1(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

if __name__ == '__main__':
    # from torchvision import models
    # model = BasicBlock(inplanes=64, planes=64)
    # model = fuknet(BasicBlock, [2, 2, 2, 2], 2)
    # print model

    model = resnext20_type1(num_classes=4)
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