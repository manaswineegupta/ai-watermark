from collections import OrderedDict
import torch
import torch.nn as nn
from math import ceil
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(
        self,
        max_res=8,
        nch=16,
        nc=3,
        bn=False,
        ws=False,
        pn=False,
        activ=nn.LeakyReLU(0.2),
    ):
        super(Generator, self).__init__()
        # resolution of output as 4 * 2^max_res: 0 -> 4x4, 1 -> 8x8, ..., 8 -> 1024x1024
        self.max_res = max_res

        # output convolutions
        self.toRGBs = nn.ModuleList()
        for i in range(self.max_res + 1):
            # max of nch * 32 feature maps as in the original article (with nch=16, 512 feature maps at max)
            self.toRGBs.append(
                conv(
                    int(nch * 2 ** (8 - max(3, i))),
                    nc,
                    kernel_size=1,
                    padding=0,
                    ws=ws,
                    activ=None,
                    gainWS=1,
                )
            )

        # convolutional blocks
        self.blocks = nn.ModuleList()
        # first block, always present
        self.blocks.append(
            nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            conv(
                                nch * 32,
                                nch * 32,
                                kernel_size=4,
                                padding=3,
                                bn=bn,
                                ws=ws,
                                pn=pn,
                                activ=activ,
                            ),
                        ),
                        (
                            "conv1",
                            conv(nch * 32, nch * 32, bn=bn, ws=ws, pn=pn, activ=activ),
                        ),
                    ]
                )
            )
        )
        for i in range(self.max_res):
            nin = int(nch * 2 ** (8 - max(3, i)))
            nout = int(nch * 2 ** (8 - max(3, i + 1)))
            self.blocks.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv0",
                                conv(nin, nout, bn=bn, ws=ws, pn=pn, activ=activ),
                            ),
                            (
                                "conv1",
                                conv(nout, nout, bn=bn, ws=ws, pn=pn, activ=activ),
                            ),
                        ]
                    )
                )
            )

        self.pn = None
        if pn:
            self.pn = PixelNormLayer()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                (
                    nn.init.normal_(m.weight, 0, 1)
                    if ws
                    else nn.init.kaiming_normal_(m.weight)
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, x=None):
        # value driving the number of layers used in generation
        if x is None:
            progress = self.max_res
        else:
            progress = min(x, self.max_res)

        alpha = progress - int(progress)

        norm_input = self.pn(input) if self.pn else input

        # generating image of size corresponding to progress
        # Example : for progress going from 0 + epsilon to 1 excluded :
        # the output will be of size 8x8 as sum of 4x4 upsampled and output of convolution
        y1 = self.blocks[0](norm_input)
        y0 = y1

        for i in range(1, int(ceil(progress) + 1)):
            y1 = F.upsample(y1, scale_factor=2)
            y0 = y1
            y1 = self.blocks[i](y0)

        # converting to RGB
        y = self.toRGBs[int(ceil(progress))](y1)

        # adding upsampled image from previous layer if transitioning, i.e. when progress is not int
        if progress % 1 != 0:
            y0 = self.toRGBs[int(progress)](y0)
            y = alpha * y + (1 - alpha) * y0

        return y


def conv(
    nin,
    nout,
    kernel_size=3,
    stride=1,
    padding=1,
    layer=nn.Conv2d,
    ws=False,
    bn=False,
    pn=False,
    activ=None,
    gainWS=2,
):
    conv = layer(
        nin,
        nout,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=False if bn else True,
    )
    layers = OrderedDict()

    if ws:
        layers["ws"] = WScaleLayer(conv, gain=gainWS)

    layers["conv"] = conv

    if bn:
        layers["bn"] = nn.BatchNorm2d(nout)
    if activ:
        if activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()') and initialized here
            layers["activ"] = activ(num_parameters=1)
        else:
            layers["activ"] = activ
    if pn:
        layers["pn"] = PixelNormLayer()
    return nn.Sequential(layers)


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__


class WScaleLayer(nn.Module):
    def __init__(self, incoming, gain=2):
        super(WScaleLayer, self).__init__()

        self.gain = gain
        self.scale = (self.gain / incoming.weight[0].numel()) ** 0.5

    def forward(self, input):
        return input * self.scale

    def __repr__(self):
        return "{}(gain={})".format(self.__class__.__name__, self.gain)
