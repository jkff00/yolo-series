import torch
from torch import nn
import math

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv2dStaticSamePadding(nn.Conv2d):#对feature map进行填充 维持原图大小
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]#输入尺寸
        kh, kw = self.weight.size()[-2:]#卷积核尺寸为cout*cin*kh*kw,返回倒数两个维度
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)#返回大于该数字的最小整数值 输出尺寸
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)#stride->[]横向步长 纵向步长
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))#左右上下填充维度
        else:
            self.static_padding = nn.Identity()

class SeparableConvBlock(nn.Module):#深度可分离卷积
    """
    created by Zylo117
    """
# 传进来的in_channel都为64
    def __init__(self, in_channels, onnx_export, norm=True, activation=True):
        super(SeparableConvBlock, self).__init__()

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels,
                                                      bias=False)  # groups=in_channel数，即：输入channel=输出channel=卷积核个数
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=1,
                                                      stride=1)  # PWconv为普通的1*1卷积
        if norm:
            self.bn = nn.BatchNorm2d(num_features=in_channels, momentum=0.01, eps=1e-3)
        else:
            self.bn = nn.Identity()
        if activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        else:
            self.swish = nn.Identity()

        self.Sequential = nn.Sequential(self.depthwise_conv, self.pointwise_conv, self.bn, self.swish)
# 可分离卷积前向传播过程：先进行DW卷积，在进行PW卷积，有BN就是用BN操作，激活类似
    def forward(self, x):
        for layer in self.Sequential:
            x = layer(x)
        return x
class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    def __init__(self, kernel_size, stride, image_size=None, **kwargs):
        super().__init__(kernel_size, stride, **kwargs)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()