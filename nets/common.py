from torch import nn
import math
from mmcv.ops import DeformConv2dPack,ModulatedDeformConv2dPack


class CR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x



class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CBR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x



class CGR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CGR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.gn = nn.GroupNorm(32,out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x





#######################################################################################################


class DeconvLayer(nn.Module):

    def __init__(self, in_planes,out_planes, deconv_kernel, deconv_stride=2, deconv_pad=1,
            deconv_out_pad=0, modulate_deform=True):
        super(DeconvLayer, self).__init__()
        if modulate_deform:
            self.dcn = ModulatedDeformConv2dPack(in_planes, out_planes, kernel_size=3, padding=1, deform_groups=1)
        else:
            self.dcn = DeformConv2dPack(in_planes, out_planes, kernel_size=3, padding=1, deform_groups=1)


        self.dcn_bn = nn.BatchNorm2d(out_planes)
        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=deconv_kernel,
            stride=deconv_stride,
            padding=deconv_pad,
            output_padding=deconv_out_pad,
            bias=False,
        )
        self._deconv_init()
        self.up_bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dcn(x)
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]




class CenternetDeconv(nn.Module):
    """
    利用deformable conv + transposed conv实现上采样
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, channels, deconv_kernel, modulate_deform):
        super(CenternetDeconv, self).__init__()

        self.deconv1 = DeconvLayer(
            channels[0], channels[1],
            deconv_kernel=deconv_kernel[0],
            modulate_deform=modulate_deform,
        )
        self.deconv2 = DeconvLayer(
            channels[1], channels[2],
            deconv_kernel=deconv_kernel[1],
            modulate_deform=modulate_deform,
        )
        self.deconv3 = DeconvLayer(
            channels[2], channels[3],
            deconv_kernel=deconv_kernel[2],
            modulate_deform=modulate_deform,
        )

    def forward(self, x):
        x = self.deconv1(x)   # 1/16
        x = self.deconv2(x)   # 1/8
        x = self.deconv3(x)   # 1/4
        return x
