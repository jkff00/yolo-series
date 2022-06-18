import math

import torch
import torch.nn as nn

from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet
from nets.Swin_transformer import Swin_transformer_Tiny

from nets.utils_extra import Swish, MemoryEfficientSwish,SeparableConvBlock, MaxPool2dStaticSamePadding, Conv2dStaticSamePadding
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', pretrained=False, input_shape=[640, 640]):
        super(YoloBody, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3 round返回四舍五入整数值
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        self.backbone_name  = backbone
        if backbone == "cspdarknet":
            #---------------------------------------------------#   
            #   生成CSPdarknet53的主干模型
            #   获得三个有效特征层，他们的shape分别是：
            #   80,80,256
            #   40,40,512
            #   20,20,1024
            #---------------------------------------------------#
            self.backbone   = CSPDarknet(base_channels, base_depth, phi, pretrained)
        else:
            #---------------------------------------------------#   
            #   如果输入不为cspdarknet，则调整通道数
            #   使其符合YoloV5的格式
            #---------------------------------------------------#
            self.backbone       = {
                'convnext_tiny'         : ConvNeXt_Tiny,
                'convnext_small'        : ConvNeXt_Small,
                'swin_transfomer_tiny'  : Swin_transformer_Tiny,
            }[backbone](pretrained=pretrained, input_shape=input_shape)
            in_channels         = {
                'convnext_tiny'         : [192, 384, 768],
                'convnext_small'        : [192, 384, 768],
                'swin_transfomer_tiny'  : [192, 384, 768],
            }[backbone]
            feat1_c, feat2_c, feat3_c = in_channels 
            self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.conv_1x1_feat3 = Conv(feat3_c, base_channels * 16, 1, 1)
            
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

        self.num_channels = 64
        self.conv_channels = [256, 512, 1024]
        self.bifpn1 = BiFPN(self.num_channels, self.conv_channels, first_time=True,onnx_export=True)
        self.bifpn2 = BiFPN(self.num_channels, self.conv_channels,onnx_export=True)
        self.sequential = nn.Sequential(self.backbone, self.bifpn1, self.bifpn2, self.bifpn2)

    def forward(self, x):
        #  backbone
        for layer in self.sequential:
            x = layer(x)
        P3, P4, P5 = x

        # if self.backbone_name != "cspdarknet":
        #     feat1 = self.conv_1x1_feat1(feat1)#进行1*1卷积
        #     feat2 = self.conv_1x1_feat2(feat2)
        #     feat3 = self.conv_1x1_feat3(feat3)
        #
        # # 20, 20, 1024 -> 20, 20, 512
        # P5          = self.conv_for_feat3(feat3)
        # # 20, 20, 512 -> 40, 40, 512
        # P5_upsample = self.upsample(P5)
        # # 40, 40, 512 -> 40, 40, 1024
        # P4          = torch.cat([P5_upsample, feat2], 1)
        # # 40, 40, 1024 -> 40, 40, 512
        # P4          = self.conv3_for_upsample1(P4)
        #
        # # 40, 40, 512 -> 40, 40, 256
        # P4          = self.conv_for_feat2(P4)
        # # 40, 40, 256 -> 80, 80, 256
        # P4_upsample = self.upsample(P4)
        # # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        # P3          = torch.cat([P4_upsample, feat1], 1)
        # # 80, 80, 512 -> 80, 80, 256
        # P3          = self.conv3_for_upsample2(P3)
        #
        # # 80, 80, 256 -> 40, 40, 256
        # P3_downsample = self.down_sample1(P3)
        # # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        # P4 = torch.cat([P3_downsample, P4], 1)
        # # 40, 40, 512 -> 40, 40, 512
        # P4 = self.conv3_for_downsample1(P4)
        #
        # # 40, 40, 512 -> 20, 20, 512
        # P4_downsample = self.down_sample2(P4)
        # # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        # P5 = torch.cat([P4_downsample, P5], 1)
        # # 20, 20, 1024 -> 20, 20, 1024
        # P5 = self.conv3_for_downsample2(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)
        return out0, out1, out2





# 定义BIFPN模块
class BiFPN(nn.Module):
    """
    modified by Zylo117
    """
# conv_channels=[40,112,320]分别对应EfficientNet中的stage4,stage6,stage8输出的特征图，即对应EfficientDet中的P3,P4,P5
# num_channels 传递进来的只有64
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        """

        Args:
            num_channels:64
            conv_channels:[40，112，320]
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
            attention：是True，则使用Fast Normalization fusion操作
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers：8个卷积操作是做特征融合后的DW卷积，保证输入输出channel一致

        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)


#up-是bifpn中TOP-down分支上特征融合过程中使用的可分离卷积，用来保证输入输出通道一致
#down-是bifpn中bottom-up分支上特征融合过程中使用的可分离卷积，同理保证输入输出的特征矩阵的channel不发生改变。
#epsilon：启用加权feature fusion时用到的参数
        # Feature scaling layers，以下是4个上采样操作，对应论文中P7in-P4in的上采样，使用最邻近插值

        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

		# 以下四个为下采样操作，对应文中P3out-P6out的下采样，使用最大池化，池化窗口3*3，步长2*2
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)


        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()#swish激活
# 如果第一次做bifpn，那么将p3,p4,p5三个特征图分别做三次普通卷积+BN操作得到P3in，P4in，P5in
        self.first_time = first_time
        if self.first_time:
       		# 对于P5而言，输入特征图的channel=320，输出channel为64得到P5in
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            # 对于P4而言，输入特征图的channel=112，输出channel为64得到P4in
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            # 对于P3而言，输入特征图的channel=40，输出channel为64得到P3in
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
			# 由于EfficientNet的backbone只将输入图像缩小为原来的1/32，对应于Efficientdet中的特征提取网络只能达到P5
			# 先要将P5进行下采样两次得到P6，P7；对于P5到P6也就是P6in，经过普通卷积+BN+maxpooling得到P7也就是P7in

			#由于P4和P5需要做一次残差连接，将P4in直接传递给下一个bifpn模块的P4，所以论文中P4->P4in,P5->P5in做了两次
			#即得到两个相同的P4in和两个相同的P5in
            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )


        # Weight 这里的4个权重为P7in上采样后乘上W1，在与P6in乘上W2，进行特征融合，初始化为两个1
        # 由于top-down分支有4次feature fusion，并且每次融合的输入有两个，并且每个权值都需要经过一次Relu，保证每个权值都≥0
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
		# 同理，对于bottom-up分支的特征融合，P4out--P6out的融合有三个输入，因此初始化三个全为1的权值，而对于P7out而言
        # 特征融合的输入只有两个，因此初始化两个全为1的权值，并且每个权值都需要经过一次Relu，保证每个权值都≥0
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.attention = attention

    def forward(self, inputs):# bifpn前向传播过程
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ?
                             ?                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------? ?
                             ?                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------? ?
                             ?                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------? ?
                             |--------------? |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            p3_out, p4_out, p5_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out = self._forward(inputs)

        return p3_out, p4_out, p5_out
	#以下为实现带有fast normalization fusion的前向传播过程
    def _forward_fast_attention(self, inputs):

# A.对于EfficientDet-D0为例，conv_channels=[40, 112, 320]列表保存的是backbone中的3个特征图的channel，也就是框架图中的P3，P4，P5的输出channel，将P3，P4以及P5经过普通1x1卷积操作将channel调整到统一的尺度为64，并且每个卷积后面都有BN操作，于是得到P3in，P4in以及P5in，此时这经过channel调整后的三个特征图才是BiFPN模块的输入。
# B.对于剩下的两个特征图怎么得到呢？将backbone中的P5经过卷积调整到channel为64，再经过BN操作，最后使用一个池化窗口为3x3步长为2的最大池化层得到P6in，作为输入到BiFPN模块的第4个预测特征图；将P6in直接经过池化窗口3x3步长为2的最大池化层得到P7in，作为输入到BiFPN模块的第5个预测特征图。
# C.针对EfficientDetB0来说，P3.shape=(batchsize, 40, 64, 64), P4.shape = (batchsize, 112,32,32),
# P5.shape = (batchsize, 320,16,16)经过1x1的普通卷积操作(输出channel均为64)+BN操作得到P3in，P4in和P5in，再将P5经过一次1*1的卷积(输出channel为64)+BN+maxpool得到P6in，
# P6in.shape=(batchsize,64,8,8),进而将P6in经过一次maxpool得到P7in，且P7in.shape=(batchsize,64, 4,4)。并且这里一共有8组权重，对应论文中的fast norm fusion，在融合的时候使用。
#
# 注意这里的每组权重都被初始化为全1，并且每个权重都经过relu激活函数，保证每个权值都是大于等于0的。并且这里的attention是一个布尔变量，为Ture就表示使用fast norm fusion机制，否则就不使用。
# D.当使用attention机制时，进入_forward_fast_attention函数中，继续判断当前的BiFPN是否是第一次进行BiFPN操作，为True则获取backbone中的3个特征图为P3，P4以及P5，将P5进行下采样得到P6in，P6in进行下采样得到P7in，P3，P4以及P5进行channel的调整得到P3in-P5in。
# 如果不是第一次进行BiFPN操作，只需要获取上个BiFPN模块的输出即P3in-P7in 5个特征图即可。

        if self.first_time:
            p3, p4, p5 = inputs



            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, = inputs



        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_in)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))
# 如果是第一次进入BIFPN模块，就会将P4in和P5in做两次相同的操作，一次用作top-down分支，一次用作bottom-up分支
        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))


        # Weights for P7_0 and P6_2 to P7_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p5_out = self.conv5_down(self.swish(weight[0] * p5_in + weight[1] * self.p5_downsample(p4_out)))

        return p3_out, p4_out, p5_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2


        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_in)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))


        # Connections for P7_0 and P6_2 to P7_2
        p5_out = self.conv5_down(self.swish(p5_in + self.p5_downsample(p4_out)))

        return p3_out, p4_out, p5_out
'''
对于第一次进行BiFPN时，在获得P3in，P4in_1, P4in_2, P5in_1, P5in_2, P6in, P7in，再将P7in进行上采样乘上权重w1与P6in*权值w2进行feature fusion得到P7up；之后再对P6up进行上采样乘上权值w1与P5in乘上权值w2做feature fusion得到P6up；再对P5up进行上采样乘上权值w1与P4in_1乘上权值w2做feature fusion得到P5up；之后对P4up进行上采样再乘上权值w1与P3in乘上权值w2融合得到P3out；注意这里的每次feature fusion之后都需要使用一个可分离卷积操作。
这里总共有4组初始化全为1的权值（size为2），即并且每个权值都经过relu激活函数，保证权值wi都≥0。并且在做特征融合的之前需要归一化（wi/w1+w2+epison）再乘上各自的input feature，也就是每融合一次进行一次加权求和。
当top-down分支操作结束后，继续判断，此时是否是第一次进行BiFPN，为True则获取P4in2和P5in2，直接将这俩个进行bottom-up分支的feature fusion，相当于残差连接，论文里面说这样可以在不增加计算成本的前提下融合更多的特征信息。
类似于top-down分支，第一组权值有3个，进行激活函数relu和归一化。Feature fusion：P4inx权值w1 + P4up x权值w2 + P3out经过下采样之后再x权值w3，融合之后使用swish激活函数并且接上一个可分离卷积得到bottom-up分支上的第一个输出P4out。
同理得到P5out和P6out，这三个feature fusion都有三个输入。
P7out的输入只有两个，downsample(P6out)和P7in，所以该节点的feature fusion只有两个权值，与P4out-P6out不同，但融合之后的操作是相同的。
最后返回的是bottom-up分支上的5个预测特征图的输出，作为下一个BiFPN的输入。
'''


# 定义类别器
class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        # conv_list是一个可分离卷积列表，长度为num_layers
        # bn_list是一个BN操作列表,长度为num_layers x len(pyramid_levels) = 3x5 = 15
        # header是最后的输出层，输出channel为num_anchor*4.
        # num_anchors表示每个特征图上的每个cell预测多少个anchor，每个anchor有4个坐标参数
        # swish为激活函数
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):  # 遍历每一个预测特征层
            for i, bn, conv in zip(range(self.num_layers), bn_list,
                                   self.conv_list):  # 对每个预测特征层上先做3个相同的DW卷积，输入输出channel都相同
                feat = conv(feat)  # 以第一个预测特征层为例，输入为(batchsize,64,h,w)-->(batchsize/4,64,h,w)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)  # 预测特征图做完3次DW卷积操作后，再进行一个卷积对所有anchor进行类别分数的预测，输出channel=num_anchors * num_classes
            # shape = (4,x*9,64,64)
            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)  # 使用contiguous变成连续的存储结构，再view，此时shape=(batch_size,h,w,9,90)
            feat = feat.contiguous().view(feat.shape[0], -1,
                                          self.num_classes)  # 得到第一个预测特征图的类别分数,shape=(batch_size,h*w*9,90)

            feats.append(feat)  # 将每一个预测特征层上的进行cls_net的输出保存到列表中

        feats = torch.cat(feats, dim=1)  # 将一个batch下的所有图片的5个预测特征层上预测的anchor全部累加起来，（5, batchsize,h*w*9,90)->shape=(5,batch_size*h*w*9,90)
        feats = feats.sigmoid()  # 对一个batch下的每张图片的所有anchors进行sigmoid处理，得到每个类别的预测分数

        return feats


'''
前向传播过程得到5个预测特征图上的所有anchor的预测类别分数，permute之后，tensor的size:[batchsize, 90x9,h, w] -> [batchsize,h, w, 9x90] 这里的9表示每个cell预测的anchor数，
90表示预测类别数，由于是使用COCO中的stuff categories所以这里类别数为90。第一个view之后的size：[batchsize, h ,w ,9, 90];
第二个view之后的size：[batchsize, hxwx9, 90];
再将一个batch下的所有图片在5个预测层上的anchors预测结果在维度1的位置上进行concat，输出的tensor.size=(batch_size, 49104, num_classes)。
最后将预测结果进行一个sigmoid函数得到每个anchors的预测类别分数。

'''
