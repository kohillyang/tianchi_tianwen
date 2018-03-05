import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

__all__ = ['xception']



class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv1d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm1d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv1d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm1d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv1d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm1d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv1d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm1d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool1d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(1, 2, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(2, 4, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        # do relu here

        self.block1 = Block(4, 8, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(8, 16, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(16, 32, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(32, 64, 2, 2, start_with_relu=True, grow_first=True)
        self.block5 = Block(64, 128, 2, 2, start_with_relu=True, grow_first=True)
        self.block6 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)
        # self.block10 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)
        # self.block11 = Block(128, 128, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv1d(256, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm1d(256)

        # do relu here
        self.conv4 = SeparableConv1d(256, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm1d(256)

        self.fc = nn.Linear(256, num_classes)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        # x = self.block10(x)
        # x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # print x.size()
        x = F.adaptive_avg_pool1d(x, (1,))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def xception(pretrained=False, **kwargs):
    """
    Construct Xception.
    """

    model = Xception(**kwargs)
    return model

if __name__ == '__main__':
    model = xception(num_classes=4)
    # print model

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