import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import inception


__all__ = ['Inception3', 'inception_v3']





def inception_v3(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return Inception3(**kwargs)


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True):
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.Conv2d_1a_3x3 = BasicConv1d(1, 2, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv1d(2, 4, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv1d(4, 6, kernel_size=3, padding=1,stride=2)
        self.Conv2d_3b_1x1 = BasicConv1d(6, 8, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv1d(8, 16, kernel_size=3, padding=1,stride=2)
        self.Conv2d_5a_3x3 = BasicConv1d(16, 32, kernel_size=3, padding=1, stride=2)
        self.Mixed_5b = InceptionA(32, pool_features=8)
        self.Mixed_5c = InceptionA(64, pool_features=16)
        self.Mixed_5d = InceptionA(72, pool_features=16)
        self.Mixed_6a = InceptionB(72)
        self.Mixed_6b = InceptionC(192, channels_7x7=32)
        self.Mixed_6c = InceptionC(192, channels_7x7=40)
        self.Mixed_6d = InceptionC(192, channels_7x7=40)
        self.Mixed_6e = InceptionC(192, channels_7x7=48)
        if aux_logits:
            self.AuxLogits = InceptionAux(192, num_classes)
        self.Mixed_7a = InceptionD(192)
        self.Mixed_7b = InceptionE(320)
        self.Mixed_7c = InceptionE(488)
        self.fc = nn.Linear(488, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        # print x.size()
        x = self.Conv2d_1a_3x3(x)
        # print x.size()
        x = self.Conv2d_2a_3x3(x)
        # print x.size()
        x = self.Conv2d_2b_3x3(x)
        # print 'befor pool1'
        # print x.size()
        x = F.max_pool1d(x, kernel_size=3, stride=2)
        # print 'after pool1'
        # print x.size()
        x = self.Conv2d_3b_1x1(x)
        # print x.size()
        x = self.Conv2d_4a_3x3(x)
        # print 'before pool2'
        # print x.size()
        x = F.max_pool1d(x, kernel_size=3, stride=2)
        # print 'after pool2'
        # print x.size()
        x = self.Conv2d_5a_3x3(x)
        # print 'Conv2d_5a_3x3'
        # print x.size()
        x = self.Mixed_5b(x)
        # print 'Mixed_5b'
        # print x.size()
        x = self.Mixed_5c(x)
        # print 'Mixed_5c'
        # print x.size()
        x = self.Mixed_5d(x)
        # print 'Mixed_5d'
        # print x.size()
        x = self.Mixed_6a(x)
        # print 'Mixed_6a'
        # print x.size()
        x = self.Mixed_6b(x)
        # print 'Mixed_6b'
        # print x.size()
        x = self.Mixed_6c(x)
        # print 'Mixed_6c'
        # print x.size()
        x = self.Mixed_6d(x)
        # print 'Mixed_6d'
        # print x.size()
        x = self.Mixed_6e(x)
        # print 'Mixed_6e'
        # print x.size()
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)

        x = self.Mixed_7a(x)
        # print 'Mixed_7a'
        # print x.size()
        x = self.Mixed_7b(x)
        # print 'Mixed_7b'
        # print x.size()
        x = self.Mixed_7c(x)
        # print 'Mixed_7c'
        # print x.size()
        x = F.adaptive_avg_pool1d(x, output_size=(1,))
        # print 'avg pool'
        # print x.size()
        x = F.dropout(x, training=self.training)
        # print x.size()
        x = x.view(x.size(0), -1)
        # print x.size()
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv1d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = BasicConv1d(in_channels, 12, kernel_size=1)
        self.branch5x5_2 = BasicConv1d(12, 16, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv1d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = BasicConv1d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv1d(in_channels, 96, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv1d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(24, 24, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv1d(in_channels, 48, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv1d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv1d(c7, c7, kernel_size=(1, ), padding=(0, ))
        self.branch7x7_3 = BasicConv1d(c7, 48, kernel_size=(7, ), padding=(3, ))

        self.branch7x7dbl_1 = BasicConv1d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv1d(c7, c7, kernel_size=(7, ), padding=(3, ))
        self.branch7x7dbl_3 = BasicConv1d(c7, c7, kernel_size=(1, ), padding=(0, ))
        self.branch7x7dbl_4 = BasicConv1d(c7, c7, kernel_size=(7, ), padding=(3, ))
        self.branch7x7dbl_5 = BasicConv1d(c7, 48, kernel_size=(1, ), padding=(0, ))

        self.branch_pool = BasicConv1d(in_channels, 48, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv1d(in_channels, 48, kernel_size=1)
        self.branch3x3_2 = BasicConv1d(48, 80, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv1d(in_channels, 48, kernel_size=1)
        self.branch7x7x3_2 = BasicConv1d(48, 48, kernel_size=(1, ), padding=(0, ))
        self.branch7x7x3_3 = BasicConv1d(48, 48, kernel_size=(7, ), padding=(3, ))
        self.branch7x7x3_4 = BasicConv1d(48, 48, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv1d(in_channels, 80, kernel_size=1)

        self.branch3x3_1 = BasicConv1d(in_channels, 96, kernel_size=1)
        self.branch3x3_2a = BasicConv1d(96, 96, kernel_size=(1, ), padding=(0, ))
        self.branch3x3_2b = BasicConv1d(96, 96, kernel_size=(3, ), padding=(1, ))

        self.branch3x3dbl_1 = BasicConv1d(in_channels, 112, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(112, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv1d(96, 96, kernel_size=(1, ), padding=(0, ))
        self.branch3x3dbl_3b = BasicConv1d(96, 96, kernel_size=(3, ), padding=(1, ))

        self.branch_pool = BasicConv1d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv1d(in_channels, 32, kernel_size=1)
        self.conv1 = BasicConv1d(32, 192, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(192, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.adaptive_avg_pool1d(x, output_size=(5,))
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)



if __name__ == '__main__':
    # from torchvision import models
    # model = BasicBlock(inplanes=64, planes=64)
    # model = fuknet(BasicBlock, [2, 2, 2, 2], 2)
    # print model

    model = inception_v3(num_classes=4)
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

    # print y.size()