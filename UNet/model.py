import torch
import torch.nn as nn
import torch.nn.functional as F


class conv3x3(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(outchannels, outchannels, 3, padding=1),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv3x3_out(nn.Module):
    def __init__(self, inchannels,midchannels, n_channels):
        super(conv3x3_out, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, midchannels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(midchannels, midchannels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(midchannels, n_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv3x3(inC, outC),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        self.upStep = nn.ConvTranspose2d(inC, outC, 2, stride=2)
        if withReLU:
            self.conv = conv3x3(inC, outC)
        else:
            self.conv = conv3x3_out(inC, outC, n_channels=3)

    def forward(self, x, x_down):
        x = self.upStep(x)
        x = torch.cat((x_down, x), dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.inputs = conv3x3(4, 64)
        self.downStep64  = downStep(64, 128)
        self.downStep128 = downStep(128, 256)
        self.downStep256 = downStep(256, 512)
        self.downStep512 = downStep(512, 1024)
        self.upStep1024 = upStep(1024, 512)
        self.upStep512 = upStep(512, 256)
        self.upStep256 = upStep(256, 128)
        self.upStep128 = upStep(128, 64, withReLU=False)

    def forward(self, x):
        InputsL = self.inputs(x)
        DownL1 = self.downStep64(InputsL)
        DownL2 = self.downStep128(DownL1)
        DownL3 = self.downStep256(DownL2)
        DownL4 = self.downStep512(DownL3)
        UpL4 = self.upStep1024(DownL4, DownL3)
        UpL3 = self.upStep512(UpL4, DownL2)
        UpL2 = self.upStep256(UpL3, DownL1)
        OutputsL = self.upStep128(UpL2, InputsL)
        return OutputsL