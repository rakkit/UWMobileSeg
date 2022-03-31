import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models
from mobileNet import MobileNet,ConvBNReLU,InvertedResidual
import numpy as np

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class _SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, scale=1, norm_type=None,psp_size=(1,3,6,8)):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.key_channels = in_channels
        self.value_channels = in_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        
        self.W =  nn.Sequential(nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm2d(self.key_channels),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5))

        self.f_query = self.f_key
        self.psp = PSPModule(psp_size)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x)
        key = self.psp(key)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, key.permute(0, 2, 1))
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        return context
    
    
class UpSample(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels,scale_factor=2):
        super().__init__()


        self.inConv = nn.Sequential(InvertedResidual(in_channels,mid_channels,1,1),
                                    nn.Dropout(0.5))
        self.outConv = nn.Sequential(
            nn.Upsample(scale_factor = scale_factor,mode="bilinear",align_corners=False),
            InvertedResidual(mid_channels,out_channels,1,1),
            nn.Dropout(0.5))

    def forward(self,x):
        x = self.inConv(x)
        x = self.outConv(x)
        return x


class chanelAtt(nn.Module):
    def __init__(self,in_channels):
        super().__init__()

        self.attConv = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, in_channels*2, 1,),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*2, in_channels, 1,),
            nn.Sigmoid()
        )
        self.Dropout = nn.Dropout(0.5)
    def forward(self,x):
        x = self.attConv(x)*x+x
        return self.Dropout(x)


class UWMobileSeg(nn.Module):
    def __init__(self,n_out):
        super().__init__()
        
        self.encoder = MobileNet()

        self.Att32 =chanelAtt(320)
        self.Conv32to4 = UpSample(320,96,24,8,)

        self.Att16 = _SelfAttentionBlock(96)
        self.Conv16to8 = UpSample(96,96,32,2)
        
        self.Att8 = _SelfAttentionBlock(32)
        self.Conv8to4 = UpSample(32*2,32*2,24,2)
        
        self.Att4 = _SelfAttentionBlock(24)

        self.Conv4 = InvertedResidual(24*3,12,1,1)

        self.outConv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear",align_corners=False),
            nn.Conv2d(12,n_out,1,1),
        )
        
        
    def forward(self,x):
        # 1/32, 1/16, 1/8, 1/4 resolution
        f32,f16,f8,f4 = self.encoder(x)
        
        f32 = self.Att32(f32)
        f32to4 = self.Conv32to4(f32)
                    
        f16 = self.Att16(f16)
        f16to8 = self.Conv16to8(f16)
        
        f8 = self.Att8(f8)
        f8 = torch.cat((f16to8,f8),1)
        f8to4 = self.Conv8to4(f8)
    
        f4 = self.Att4(f4)
        f4 = torch.cat((f4,f8to4,f32to4),1)        
        f4 = self.Conv4(f4)
        
        out = self.outConv(f4)
    
        return out
        
        
if __name__ == "__main__":
    net = UWMobileSeg(3).eval()
    inputTensor = torch.randn(1,3,320,320)
    out = net(inputTensor)
    print(out.shape)