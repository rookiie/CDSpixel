from matplotlib.pyplot import axis
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .model_util import *
from train_util import *

# define the function includes in import *
__all__ = [
    'CDSNet'
]

class Disentangle(nn.Module):
    def __init__(self, dim=32): # when embedderv3, this is 64
        super(Disentangle, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(dim, dim, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        unique = x * (1 - y).expand_as(x)
        common = x * y.expand_as(x)
        
        return unique, common



class CDSpixelNet(nn.Module):
    expansion = 1

    def __init__(self, bn=True, grid_size=16, use_assist=True):
        super(CDSpixelNet,self).__init__()

        self.bn = bn
        self.assign_ch = 9
        self.use_assist = use_assist
        if self.use_assist and self.training:
            self.encoder_assit = Embedder()
            self.gap = nn.AdaptiveAvgPool2d((1))            
            
        self.encoder = Embedder() 
        self.MI = Disentangle()
        self.decoder = Diffusion(grid_size=grid_size)
        self.head = nn.Sequential(
            predict_mask(32, self.assign_ch),
            nn.Softmax(1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x, x_assit=None, spixel=None):
        if self.training:
            assert x_assit is not None, "Training use input with two modal."
            return self._forward_train(x, x_assit, spixel)
        else:
            return self._forward_inference(x, spixel)
    
    def _forward_train(self, x, x_assit=None, spixel=None):
        if self.use_assist:
            x_assit = self.encoder_assit(x_assit)
            x_assit, mi_assit = self.MI(x_assit)
            flow_assit = self.decoder(x_assit, spixel)
            prob_assit = self.head(flow_assit)
            
            x = self.encoder(x)
            x, mi = self.MI(x)
            flow = self.decoder(x, spixel)    
            prob = self.head(flow)
            return prob, prob_assit, [x, x_assit], [self.gap(mi), self.gap(mi_assit)]
        else:
            x = self.encoder(x)
            flow = self.decoder(x, spixel)    
            prob = self.head(flow)
            return prob
        
    
    def _forward_inference(self, x, spixel=None):
        x = self.encoder(x)
        if self.use_assist:
            x, _ = self.MI(x)
        flow = self.decoder(x, spixel)
        prob = self.head(flow)        
        return prob
    
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
        


def CDSNet(data=None):
    # model with batch normalization
    model = CDSpixelNet(bn=True)
    if data is not None:
        model.load_state_dict(data['state_dict'],strict=False)
    return model
#
