import torch.nn as nn


class resBlock(nn.Module):
    def __init__(self, inchannel, outchannel, downsample=False, upsample=False):
        super(resBlock, self).__init__()
        self.downsample = downsample
        self.upsample = upsample
        self.reflectpadding1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv0down = nn.Conv2d(inchannel, outchannel,kernel_size=1,stride=2,padding=0, bias=False)
        self.conv0up = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1,stride=1,padding=0, bias=False),
            nn.Upsample(scale_factor=2),
        )
        if downsample:
            self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2,padding=0, bias=False)
        elif upsample:
            self.conv1 = nn.ConvTranspose2d(inchannel, outchannel, kernel_size=3, stride=2, padding=3, output_padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1,padding=0, bias=False)
        self.bn = nn.BatchNorm2d(outchannel, track_running_stats=False)
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(outchannel, outchannel, kernel_size=3,stride=1,padding=0, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=False)
        )
        self.finalNorm = nn.BatchNorm2d(outchannel, track_running_stats=False)

    def forward(self, x):
        identity = x
        out = self.reflectpadding1(x)
        convout = self.conv1(out)
        convout = self.bn(convout)
        mainout = self.conv2(convout)
        if self.downsample:
            sideout = self.conv0down(identity)
        elif self.upsample:
            sideout = self.conv0up(identity)
        else:
            sideout = identity
        finalout_ = sideout + mainout
        finalout = self.finalNorm(finalout_)
        return finalout

class resNet(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(resNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d((3,3,3,3)),
            nn.Conv2d(inchannel, 32, kernel_size=7,stride=1,padding=0,bias=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((2,2,2,2)),
            nn.Conv2d(32, 64, kernel_size=5, stride=2,padding=0,bias=False),
            nn.ReLU(inplace=True),
        )
        self.downsampleblock = resBlock(inchannel=64, outchannel=128, downsample=True)
        self.standblock = resBlock(inchannel=128, outchannel=128)
        self.upsampleblock = resBlock(inchannel=128, outchannel=64, upsample=True)
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)),
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=5,stride=2,padding=4,output_padding=1, bias=False),
            nn.ReLU(),
            nn.ReflectionPad2d((3,3,3,3)),
            nn.Conv2d(in_channels=32, out_channels=outchannel, kernel_size=7, stride=1,padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.downsampleblock(out)
        for i in range(6):
            out = self.standblock(out)
        out = self.upsampleblock(out)
        out = self.conv2(out)
        return out
