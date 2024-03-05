from audioop import bias
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
## *************************** my functions ****************************

def predict_param(in_planes, channel=3):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

def predict_mask(in_planes, channel=9):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

def predict_feat(in_planes, channel=20, stride=1):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=stride, padding=1, bias=True)

def predict_prob(in_planes, channel=9):
    return  nn.Sequential(
        nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Softmax(1)
    )
#***********************************************************************

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=None):
    if padding is None:
        padding = (kernel_size-1)//2
    else:
        padding = padding
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.1)
        )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )


class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.conv0a = conv(True, 3, 64, kernel_size=3)
        self.conv0b = conv(True, 64, 64, kernel_size=3)

        self.pool0 = nn.MaxPool2d(3, 2, 1)

        self.conv1a = conv(True, 64, 64, kernel_size=3)
        self.conv1b = conv(True, 64, 64, kernel_size=3)

        self.pool1 = nn.MaxPool2d(3, 2, 1)

        self.conv2a = conv(True, 64, 64, kernel_size=3)
        self.conv2b = conv(True, 64, 64, kernel_size=3)

        self.head0 = conv(True, 64 * 3, 32, kernel_size=3)
        self.head1 = nn.Sequential(
            nn.Conv2d(32 + 3, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )
        
    def forward(self, x):
        x0 = self.conv0b(self.conv0a(x))
        x1 = self.pool0(x0)
        x1 = self.conv1b(self.conv1a(x1))
        x2 = self.pool1(x1)
        x2 = self.conv2b(self.conv2a(x2))

        x1_up = F.interpolate(x1, scale_factor=2)
        x2_up = F.interpolate(x2, scale_factor=4)

        out = torch.cat([x0, x1_up, x2_up], 1)
        out = self.head0(out)
        out = self.head1(torch.cat([out, x], dim=1))
        return out

class water_diffusion(nn.Module):
    def __init__(self, dim=32):
        super(water_diffusion, self).__init__()
        self.conv_up = deconv(dim, dim) # up ->2
        self.fusion = conv(True, dim * 2, dim, kernel_size=3)
            
    def forward(self, input, flow):
        flow = self.conv_up(flow)
        _, _, h,w = flow.shape
        inp = F.interpolate(input, size=(h,w), mode='bilinear')
        out = self.fusion(torch.cat([inp, flow], dim=1))
        return out        


class Diffusion(nn.Module):
    def __init__(self, grid_size):
        super(Diffusion, self).__init__()
        self.diffusion_step = int(np.log2(grid_size))
        self.up = nn.ModuleList([water_diffusion() for i in range(self.diffusion_step)])
    
    def forward(self, x, spixel=None):
        b,c,h,w = x.shape    
        if spixel is None:
            spixel = (h//16 , w//16)
        flow = F.interpolate(x, size=spixel, mode='bilinear')    
        for stage in self.up:
            flow = stage(x, flow)
        return flow