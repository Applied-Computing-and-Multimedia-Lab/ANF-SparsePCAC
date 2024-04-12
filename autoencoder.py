import torch
import MinkowskiEngine as ME
from functional import bound

from data_utils import isin, istopk


def make_layer(block, block_layers, channels):
    """make stacked InceptionResNet layers.
    """
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))

    return torch.nn.Sequential(*layers)

class get_coordinate(torch.nn.Module):
    def __init__(self):
        super(get_coordinate, self).__init__()
        self.SumPooling = ME.MinkowskiSumPooling(kernel_size=2, stride=2, dilation=1, dimension=3)

    def forward(self, input):
        coordinate1 = self.SumPooling(input)
        coordinate2 = self.SumPooling(coordinate1)
        coordinate3 = self.SumPooling(coordinate2)

        return coordinate2, coordinate3

class Encoder(torch.nn.Module):
    def __init__(self, channels=[3, 64, 128]):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # no IRN
        out0 = self.relu(self.down0(self.conv0(x)))
        out1 = self.relu(self.down1(self.conv1(out0)))
        out2 = self.down2(self.conv2(out1))
        out_cls_list = [out2, out1, out0]

        return out_cls_list


class Decoder(torch.nn.Module):
    """the decoding network with upsampling.
    """

    def __init__(self, channels=[3, 64, 128]):
        super().__init__()
        self.up0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.up1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.up2 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[1],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)


    def forward(self, x):
        # no IRN
        out2 = self.relu(self.conv0(self.up0((x))))
        out1 = self.relu(self.conv1(self.up1(out2)))
        out0 = self.conv2(self.up2(out1))

        out_cls_list = [out2, out1, out0]

        return out_cls_list, out0


class HPEncoder(torch.nn.Module):
    def __init__(self, channels=[128, 128, 128]):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[1],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out0 = self.relu(self.conv0(x))
        out1 = self.relu(self.down1(self.conv1(out0)))
        out2 = self.down2(self.conv2(out1))

        return out2


class HPDecoder(torch.nn.Module):
    """the decoding network with upsampling.
    """

    def __init__(self, channels=[256, 256, 128]):
        super().__init__()

        self.up0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.up1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[1],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out0 = self.relu(self.conv0(self.up0(x)))
        out1 = self.relu(self.conv1(self.up1(out0)))
        out2 = self.conv2(out1)

        return out2


class PC_DQ_ResBlock(torch.nn.Module):
    """Inception Residual Network
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiLeakyReLU(0.2, inplace=True)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

    def forward(self, input):
        out = self.conv2(self.relu(self.conv1(input)))
        return out + input


class PC_DeQuantizationModule(torch.nn.Module):
    def __init__(self, channels=[1, 16, 1], num_layers=6):
        super().__init__()

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.resblock = make_layer(
            block=PC_DQ_ResBlock,
            block_layers=num_layers,
            channels=channels[1])
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

    def forward(self, input):
        conv1 = self.conv1(input)
        x = self.resblock(conv1)
        conv2 = self.conv2(x) + conv1
        conv3 = self.conv3(conv2) + input

        return conv3