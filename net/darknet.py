import torch
import torch.nn as nn
import math
from collections import OrderedDict


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv3d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu2(out)

        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()


        self.inplanes = 32

        # 卷积 1->32  128 x 128
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)
        # 1个 参差  #32->64 64 x 64

        self.layer1 = self._make_layer([32, 64], layers[0])

        # 2个 参差

        self.layer2 = self._make_layer([64, 128], layers[1])

        # 8个 参差
        self.layer3 = self._make_layer([128, 256], layers[2])

        # 4个 参差
        self.layer4 = self._make_layer([256, 512], layers[3])

        self.layers_out_filters = [64, 128, 256, 512]

        # 进行权值初始化   去掉初始化
        #for m in self.modules():
            #if isinstance(m, nn.Conv3d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
               # m.weight.data.normal_(0, math.sqrt(2. / n))
          #  elif isinstance(m, nn.BatchNorm3d):
            #    m.weight.data.fill_(1)
          #     m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        # 第一次  32->64   64*64
        layers.append(("ds_conv", nn.Conv3d(self.inplanes, planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm3d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入darknet模块

        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)

        return out3, out4

def darknet_vessel():
    model = DarkNet([1, 2, 4, 2])
    return model
