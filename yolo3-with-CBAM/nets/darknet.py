import math
from collections import OrderedDict

import torch
import torch.nn as nn


#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1  = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(planes[0])
        self.relu1  = nn.LeakyReLU(0.1)
        
        self.conv2  = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes[1])
        self.relu2  = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out

class Channel_Attention(nn.Module):#CAM
    def __init__(self,C1,ratio=16):
        super(Channel_Attention, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)#1*1*C
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Conv2d(C1, C1/ratio,bias=False)
        self.fc2 = nn.Conv2d(C1/ratio, C1,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,X):
        avg_out = self.fc2(self.relu(self.fc1(self.avgpool(X))))
        max_out = self.fc2(self.relu(self.fc1(self.maxpool(X))))
        out = avg_out + max_out
        return X*self.sigmoid(out)
class Spatial_Attention(nn.Module):
    def __init__(self,):
        super(Spatial_Attention, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)#对最后一维度进行池化操作
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)#基于通道的全局池化，channel->1

        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(2,1,kernel_size=7)
        self.concat = torch.cat()

    def forward(self,X):#输入尺寸H*W*C
        avg_out = self.global_avgpool(X)
        max_out = self.global_maxpool(X)
        return X*(self.sigmoid(self.conv(self.concat( [avg_out, max_out], -1 ))) )

class CBAM_Block(nn.Module):#由于广播机制 输出尺寸与输入尺寸一致
    def __init__(self,C1):
        super(CBAM_Block, self).__init__()
        self.channel_attention = Channel_Attention(C1)
        self.spatial_Attention = Spatial_Attention()
    def forward(self, X):
        channel_out = self.channel_attention(X)
        spatial_out = self.spatial_Attention(channel_out)
        return spatial_out




class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1  = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(self.inplanes)
        self.relu1  = nn.LeakyReLU(0.1)
        self.cbam1 = CBAM_Block(self.inplanes)#CBAM加在网络开始与结尾处 不改变网络结构 可以加载预训练权重

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0])
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.cbam1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model
