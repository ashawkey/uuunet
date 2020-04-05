# this is where magic happens.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from batchgenerators.utilities.file_and_folder_operations import *
from uuunet.network_architecture.neural_network import SegmentationNetwork
from time import time
import numpy as np


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        # x: [B, C, H, W, D]
        s = F.adaptive_avg_pool2d(x, 1) # [B, C, 1, 1, 1]
        s = self.conv1(s) # [B, C//reduction, 1, 1, 1]
        s = F.relu(s, inplace=True)
        s = self.conv2(s) # [B, C, 1, 1, 1]
        x = x + torch.sigmoid(s)
        return x

class SEModule3D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels//reduction, kernel_size=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels//reduction, in_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        # x: [B, C, H, W, D]
        s = F.adaptive_avg_pool3d(x, 1) # [B, C, 1, 1, 1]
        s = self.conv1(s) # [B, C//reduction, 1, 1, 1]
        s = F.relu(s, inplace=True)
        s = self.conv2(s) # [B, C, 1, 1, 1]
        x = x + torch.sigmoid(s)
        return x

class ConvBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, dilation=1, stride=1, groups=1, is_activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.is_activation = is_activation
        
        if is_activation:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.is_activation:
            x = self.relu(x)
        return x

class ConvBR3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, dilation=1, stride=1, groups=1, is_activation=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, groups=groups, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.is_activation = is_activation
        
        if is_activation:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.is_activation:
            x = self.relu(x)
        return x


class SENextBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None, groups=32, reduction=16, pool=None, is_shortcut=False):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBR2d(in_channels, mid_channels, 1, 0, 1, )
        self.conv2 = ConvBR2d(mid_channels, mid_channels, 3, 1, 1, groups=groups)
        self.conv3 = ConvBR2d(mid_channels, out_channels, 1, 0, 1, is_activation=False)
        self.se = SEModule(out_channels, reduction)
        self.stride = stride
        self.is_shortcut = is_shortcut
        
        if is_shortcut:
            self.shortcut = ConvBR2d(in_channels, out_channels, 1, 0, 1, is_activation=False)
        if stride is not None:
            if pool == 'max':
                self.pool = nn.MaxPool2d(stride, stride)
            elif pool == 'avg':
                self.pool = nn.AvgPool2d(stride, stride)
    
    def forward(self, x):
        s = self.conv1(x)
        s = self.conv2(s)
        if self.stride is not None:
            s = self.pool(s)
        s = self.conv3(s)
        s = self.se(s)
        
        if self.is_shortcut:
            if self.stride is not None:
                x = F.avg_pool2d(x, self.stride, self.stride) # avg
            x = self.shortcut(x)
        
        x = x + s
        x = F.relu(x, inplace=True)
        
        return x

class SENextBottleneck3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None, groups=32, reduction=16, pool=None, is_shortcut=False):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBR3d(in_channels, mid_channels, 1, 0, 1, )
        self.conv2 = ConvBR3d(mid_channels, mid_channels, 3, 1, 1, groups=groups)
        self.conv3 = ConvBR3d(mid_channels, out_channels, 1, 0, 1, is_activation=False)
        self.se = SEModule3D(out_channels, reduction)
        self.stride = stride
        self.is_shortcut = is_shortcut
        
        if is_shortcut:
            self.shortcut = ConvBR3d(in_channels, out_channels, 1, 0, 1, is_activation=False)
        if stride is not None:
            if pool == 'max':
                self.pool = nn.MaxPool3d(stride, stride)
            elif pool == 'avg':
                self.pool = nn.AvgPool3d(stride, stride)
    
    def forward(self, x):
        s = self.conv1(x)
        s = self.conv2(s)
        if self.stride is not None:
            s = self.pool(s)
        s = self.conv3(s)
        s = self.se(s)
        
        if self.is_shortcut:
            if self.stride is not None:
                x = F.avg_pool3d(x, self.stride, self.stride) # avg
            x = self.shortcut(x)
        
        x = x + s
        x = F.relu(x, inplace=True)
        
        return x

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

    
class Attention(nn.Module):
    def __init__(self, in_channels, skip_channels, int_channels):
        super().__init__()
        self.att1 = ConvBR2d(in_channels, int_channels, 1, 0, is_activation=False)
        self.att2 = ConvBR2d(skip_channels, int_channels, 1, 0, is_activation=False)
        self.att3 = ConvBR2d(int_channels, 1, 1, 0, is_activation=False)

    def forward(self, g, x):
        g1 = self.att1(g)
        x1 = self.att2(x)
        psi = F.relu(g1+x1, inplace=True)
        psi = F.sigmoid(self.att3(psi))

        return x*psi

class SparseAttBlock(nn.Module):
    def __init__(self, fin, fout, stride=(16, 16), fmid=None):
        super().__init__()
        
        self.stride = stride

        if fmid is None:
            fmid = fin // 2

        self.att1 = DotProductAttBlock(fin, fout, fmid)
        self.att2 = DotProductAttBlock(fin, fout, fmid)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """

        B, C, H, W = x.shape
        h1, w1 = self.stride
        h2, w2 = H//h1, W//w1

        x = x.view(B, C, h1, h2, w1, w2)
        
        # long
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B*h2*w2, C, h1, w1)
        x = self.att1(x)
        x = x.view(B, h2, w2, C, h1, w1)

        # short
        x = x.permute(0, 4, 5, 3, 1, 2).contiguous()
        x = x.view(B*h1*w1, C, h2, w2)
        x = self.att2(x)
        x = x.view(B, h1, w1, C, h2, w2)

        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)

        return x

class DotProductAttBlock(nn.Module):
    def __init__(self, fin, fout, fmid=None):
        super().__init__()

        if fmid is None:
            fmid = fin // 2

        self.fin = fin
        self.fout = fout
        self.fmid = fmid

        self.convx = nn.Conv2d(fin, fmid, 1, 1, 0)
        self.convy = nn.Conv2d(fin, fmid, 1, 1, 0)

        self.conv = nn.Sequential(
                nn.Conv2d(fin, fout, 1, 1, 0),
                nn.BatchNorm2d(fout),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, y=None):
        """
        x, y: [B, C, H, W]
        """

        B, C, H, W = x.shape

        if y is None:
            y = x
        
        theta = self.convx(x).view(B, self.fmid, -1).permute(0,2,1).contiguous()
        psi = self.convy(y).view(B, self.fmid, -1)
        alpha = theta @ psi # [B, HW, HW]
        alpha = F.softmax(alpha, 2)
        
        x = x.view(B, C, -1).permute(0,2,1).contiguous() # [B, HW, C]
        x = alpha @ x # [B, HW, C]
        x = x.permute(0,2,1).contiguous().view(B, C, H, W)

        x = self.conv(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.conv1 = ConvBR2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBR2d(out_channels, out_channels, kernel_size=3, padding=1)
        # att
        #self.att = Attention(in_channels, skip_channels, skip_channels//2)
        
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=self.stride, mode="bilinear")
        #skip = self.att(x, skip) # gate skip with x
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.conv1 = ConvBR3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBR3d(out_channels, out_channels, kernel_size=3, padding=1)
        # att
        #self.att = Attention(in_channels, skip_channels, skip_channels//2)
        
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=self.stride, mode="trilinear")
        #skip = self.att(x, skip) # gate skip with x
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class X3Net(nn.Module):
    def __init__(self, 
                 num_features=1,
                 ):
        super().__init__()
        
        backbone = [1,1,1,1]
        encoder_channels = [32, 64, 128, 256, 512]
        decoder_channels = [256, 128, 64, 32]
        strides = [[2,2], [2,2], [2,2], [2,2]]

        ### MR encoder [256, 256]
        self.block00 = nn.Sequential(
            ConvBR2d(num_features, encoder_channels[0], kernel_size=5, stride=1, padding=2),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
        )
        self.block01 = nn.Sequential(
            SENextBottleneck(encoder_channels[0], encoder_channels[1], stride=strides[0], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[1], encoder_channels[1], stride=None, is_shortcut=False) for i in range(backbone[0])]
        )

        ### shared
        self.block2 = nn.Sequential(
            SENextBottleneck(encoder_channels[1], encoder_channels[2], stride=strides[1], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[2], encoder_channels[2], stride=None, is_shortcut=False) for i in range(backbone[1])]
        )
        self.block3 = nn.Sequential(
            SENextBottleneck(encoder_channels[2], encoder_channels[3], stride=strides[2], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[3], encoder_channels[3], stride=None, is_shortcut=False) for i in range(backbone[2])]
        )
        self.block4 = nn.Sequential(
            SENextBottleneck(encoder_channels[3], encoder_channels[4], stride=strides[3], is_shortcut=True, pool='avg'),
          *[SENextBottleneck(encoder_channels[4], encoder_channels[4], stride=None, is_shortcut=False) for i in range(backbone[3])]
        )  
        self.deblock4 = DecoderBlock(encoder_channels[-1], encoder_channels[-2], decoder_channels[0], strides[-1])
        self.deblock3 = DecoderBlock(decoder_channels[0], encoder_channels[-3], decoder_channels[1], strides[-2])

        ### MR decoder
        self.deblock02 = DecoderBlock(decoder_channels[1], encoder_channels[-4], decoder_channels[2], strides[-3])
        self.deblock01 = DecoderBlock(decoder_channels[2], encoder_channels[-5], decoder_channels[3], strides[-4])

        ### MR seg head
        self.head0 = nn.Conv2d(decoder_channels[-1], 5, 1, 1, 0, bias=False)


        ### CT encoder [512, 512]
        self.block10 = nn.Sequential(
            ConvBR2d(num_features, encoder_channels[0], kernel_size=5, stride=1, padding=2),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
        )
        self.block11 = nn.Sequential(
            SENextBottleneck(encoder_channels[0], encoder_channels[1], stride=strides[0], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[1], encoder_channels[1], stride=None, is_shortcut=False) for i in range(backbone[0])]
        )

        ### CT decoder
        self.deblock12 = DecoderBlock(decoder_channels[1], encoder_channels[-4], decoder_channels[2], strides[-3])
        self.deblock11 = DecoderBlock(decoder_channels[2], encoder_channels[-5], decoder_channels[3], strides[-4])

        ### CT seg head
        self.head1 = nn.Conv2d(decoder_channels[-1], 2, 1, 1, 0, bias=False)

        ### Attention
        self.att0 = SparseAttBlock(decoder_channels[-1], decoder_channels[-1])

    def forward(self, x, modality):
        if modality == 0:
            x0 = self.block00(x)
            x1 = self.block01(x0)

            x2 = self.block2(x1)
            x3 = self.block3(x2)
            x4 = self.block4(x3)
            x = self.deblock4(x4, x3)
            x = self.deblock3(x, x2)

            x = self.deblock02(x, x1)
            x = self.deblock01(x, x0)
            x = self.att0(x)
            x = self.head0(x)
        else:
            x = F.interpolate(x, size=(256, 256), mode='area')
            x0 = self.block10(x)
            x1 = self.block11(x0)
            
            x2 = self.block2(x1)
            x3 = self.block3(x2)
            x4 = self.block4(x3)
            x = self.deblock4(x4, x3)
            x = self.deblock3(x, x2)
            
            x = self.deblock12(x, x1)
            x = self.deblock11(x, x0)
            x = self.head1(x)            
            x = F.interpolate(x, size=(512, 512), mode='area')
        return x

class X2Net(nn.Module):
    def __init__(self, 
                 num_features=1,
                 ):
        super().__init__()
        
        backbone = [1,1,1,1]
        encoder_channels = [32, 64, 128, 256, 512]
        decoder_channels = [256, 128, 64, 32]
        strides = [[2,2], [2,2], [2,2], [2,2]]

        ### MR encoder [256, 256]
        self.block00 = nn.Sequential(
            ConvBR2d(num_features, encoder_channels[0], kernel_size=5, stride=1, padding=2),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
        )
        self.block01 = nn.Sequential(
            SENextBottleneck(encoder_channels[0], encoder_channels[1], stride=strides[0], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[1], encoder_channels[1], stride=None, is_shortcut=False) for i in range(backbone[0])]
        )
        self.block02 = nn.Sequential(
            SENextBottleneck(encoder_channels[1], encoder_channels[2], stride=strides[1], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[2], encoder_channels[2], stride=None, is_shortcut=False) for i in range(backbone[1])]
        )

        ### shared
        self.block3 = nn.Sequential(
            SENextBottleneck(encoder_channels[2], encoder_channels[3], stride=strides[2], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[3], encoder_channels[3], stride=None, is_shortcut=False) for i in range(backbone[2])]
        )
        self.block4 = nn.Sequential(
            SENextBottleneck(encoder_channels[3], encoder_channels[4], stride=strides[3], is_shortcut=True, pool='avg'),
          *[SENextBottleneck(encoder_channels[4], encoder_channels[4], stride=None, is_shortcut=False) for i in range(backbone[3])]
        )  
        self.deblock4 = DecoderBlock(encoder_channels[-1], encoder_channels[-2], decoder_channels[0], strides[-1])

        ### MR decoder
        self.deblock03 = DecoderBlock(decoder_channels[0], encoder_channels[-3], decoder_channels[1], strides[-2])
        self.deblock02 = DecoderBlock(decoder_channels[1], encoder_channels[-4], decoder_channels[2], strides[-3])
        self.deblock01 = DecoderBlock(decoder_channels[2], encoder_channels[-5], decoder_channels[3], strides[-4])

        ### MR seg head
        self.head0 = nn.Conv2d(decoder_channels[-1], 5, 1, 1, 0, bias=False)


        ### CT encoder [512, 512]
        self.block10 = nn.Sequential(
            ConvBR2d(num_features, encoder_channels[0], kernel_size=5, stride=1, padding=2),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
        )
        self.block11 = nn.Sequential(
            SENextBottleneck(encoder_channels[0], encoder_channels[1], stride=strides[0], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[1], encoder_channels[1], stride=None, is_shortcut=False) for i in range(backbone[0])]
        )
        self.block12 = nn.Sequential(
            SENextBottleneck(encoder_channels[1], encoder_channels[2], stride=strides[1], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[2], encoder_channels[2], stride=None, is_shortcut=False) for i in range(backbone[1])]
        )

        ### CT decoder
        self.deblock13 = DecoderBlock(decoder_channels[0], encoder_channels[-3], decoder_channels[1], strides[-2])
        self.deblock12 = DecoderBlock(decoder_channels[1], encoder_channels[-4], decoder_channels[2], strides[-3])
        self.deblock11 = DecoderBlock(decoder_channels[2], encoder_channels[-5], decoder_channels[3], strides[-4])

        ### CT seg head
        self.head1 = nn.Conv2d(decoder_channels[-1], 2, 1, 1, 0, bias=False)

    def forward(self, x, modality):
        if modality == 0:
            x0 = self.block00(x)
            x1 = self.block01(x0)
            x2 = self.block02(x1)

            x3 = self.block3(x2)
            x4 = self.block4(x3)
            x = self.deblock4(x4, x3)

            x = self.deblock03(x, x2)
            x = self.deblock02(x, x1)
            x = self.deblock01(x, x0)
            x = self.head0(x)
        else:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
            x0 = self.block10(x)
            x1 = self.block11(x0)
            x2 = self.block12(x1)

            x3 = self.block3(x2)
            x4 = self.block4(x3)
            x = self.deblock4(x4, x3)

            x = self.deblock13(x, x2)
            x = self.deblock12(x, x1)
            x = self.deblock11(x, x0)
            x = self.head1(x)            
            x = F.interpolate(x, size=(512, 512), mode='bilinear')
        return x

class X2Net3D(nn.Module):
    def __init__(self, 
                 num_features=1,
                 ):
        super().__init__()
        
        backbone = [1,1,1,1]
        encoder_channels = [32, 64, 128, 256, 512]
        decoder_channels = [256, 128, 64, 32]
        strides = [[2,2,2], [2,2,2], [2,2,2], [2,2,2]]

        ### MR encoder [256, 256]
        self.block00 = nn.Sequential(
            ConvBR3d(num_features, encoder_channels[0], kernel_size=5, stride=1, padding=2),
            ConvBR3d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR3d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
        )
        self.block01 = nn.Sequential(
            SENextBottleneck3D(encoder_channels[0], encoder_channels[1], stride=strides[0], is_shortcut=True, pool='max'),
          *[SENextBottleneck3D(encoder_channels[1], encoder_channels[1], stride=None, is_shortcut=False) for i in range(backbone[0])]
        )
        self.block02 = nn.Sequential(
            SENextBottleneck3D(encoder_channels[1], encoder_channels[2], stride=strides[1], is_shortcut=True, pool='max'),
          *[SENextBottleneck3D(encoder_channels[2], encoder_channels[2], stride=None, is_shortcut=False) for i in range(backbone[1])]
        )

        ### shared
        self.block3 = nn.Sequential(
            SENextBottleneck3D(encoder_channels[2], encoder_channels[3], stride=strides[2], is_shortcut=True, pool='max'),
          *[SENextBottleneck3D(encoder_channels[3], encoder_channels[3], stride=None, is_shortcut=False) for i in range(backbone[2])]
        )
        self.block4 = nn.Sequential(
            SENextBottleneck3D(encoder_channels[3], encoder_channels[4], stride=strides[3], is_shortcut=True, pool='avg'),
          *[SENextBottleneck3D(encoder_channels[4], encoder_channels[4], stride=None, is_shortcut=False) for i in range(backbone[3])]
        )  
        self.deblock4 = DecoderBlock3D(encoder_channels[-1], encoder_channels[-2], decoder_channels[0], strides[-1])

        ### MR decoder
        self.deblock03 = DecoderBlock3D(decoder_channels[0], encoder_channels[-3], decoder_channels[1], strides[-2])
        self.deblock02 = DecoderBlock3D(decoder_channels[1], encoder_channels[-4], decoder_channels[2], strides[-3])
        self.deblock01 = DecoderBlock3D(decoder_channels[2], encoder_channels[-5], decoder_channels[3], strides[-4])

        ### MR seg head
        self.head0 = nn.Conv3d(decoder_channels[-1], 5, 1, 1, 0, bias=False)


        ### CT encoder [512, 512]
        self.block10 = nn.Sequential(
            ConvBR3d(num_features, encoder_channels[0], kernel_size=5, stride=1, padding=2),
            ConvBR3d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR3d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
        )
        self.block11 = nn.Sequential(
            SENextBottleneck3D(encoder_channels[0], encoder_channels[1], stride=strides[0], is_shortcut=True, pool='max'),
          *[SENextBottleneck3D(encoder_channels[1], encoder_channels[1], stride=None, is_shortcut=False) for i in range(backbone[0])]
        )
        self.block12 = nn.Sequential(
            SENextBottleneck3D(encoder_channels[1], encoder_channels[2], stride=strides[1], is_shortcut=True, pool='max'),
          *[SENextBottleneck3D(encoder_channels[2], encoder_channels[2], stride=None, is_shortcut=False) for i in range(backbone[1])]
        )

        ### CT decoder
        self.deblock13 = DecoderBlock3D(decoder_channels[0], encoder_channels[-3], decoder_channels[1], strides[-2])
        self.deblock12 = DecoderBlock3D(decoder_channels[1], encoder_channels[-4], decoder_channels[2], strides[-3])
        self.deblock11 = DecoderBlock3D(decoder_channels[2], encoder_channels[-5], decoder_channels[3], strides[-4])

        ### CT seg head
        self.head1 = nn.Conv3d(decoder_channels[-1], 2, 1, 1, 0, bias=False)

    def forward(self, x, modality):
        if modality == 0:
            x0 = self.block00(x)
            x1 = self.block01(x0)
            x2 = self.block02(x1)

            x3 = self.block3(x2)
            x4 = self.block4(x3)
            x = self.deblock4(x4, x3)

            x = self.deblock03(x, x2)
            x = self.deblock02(x, x1)
            x = self.deblock01(x, x0)
            x = self.head0(x)
        else:
            original_shape = x.shape[2:]
            x = F.interpolate(x, size=(32, 256, 256), mode='area')
            x0 = self.block10(x)
            x1 = self.block11(x0)
            x2 = self.block12(x1)

            x3 = self.block3(x2)
            x4 = self.block4(x3)
            x = self.deblock4(x4, x3)

            x = self.deblock13(x, x2)
            x = self.deblock12(x, x1)
            x = self.deblock11(x, x0)
            x = self.head1(x)            
            x = F.interpolate(x, size=original_shape, mode='area')
        return x


class XNet(SegmentationNetwork):
    def __init__(self, 
                 num_features=1,
                 ):
        super().__init__()
        
        backbone = [2,2,2,2]
        encoder_channels = [64, 128, 192, 256, 512]
        decoder_channels = [256, 128, 64, 32]
        strides = [[2,2], [2,2], [2,2], [2,2]]
        # must
        self.conv_op = nn.Conv2d
        self.input_shape_must_be_divisible_by = np.prod(strides, 0, dtype=np.int64)

        ### MR encoder [256, 256]
        self.block00 = nn.Sequential(
            ConvBR2d(num_features, encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
        )
        self.block01 = nn.Sequential(
            SENextBottleneck(encoder_channels[0], encoder_channels[1], stride=strides[0], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[1], encoder_channels[1], stride=None, is_shortcut=False) for i in range(backbone[0])]
        )
        self.block02 = nn.Sequential(
            SENextBottleneck(encoder_channels[1], encoder_channels[2], stride=strides[1], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[2], encoder_channels[2], stride=None, is_shortcut=False) for i in range(backbone[1])]
        )
        self.block03 = nn.Sequential(
            SENextBottleneck(encoder_channels[2], encoder_channels[3], stride=strides[2], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[3], encoder_channels[3], stride=None, is_shortcut=False) for i in range(backbone[2])]
        )
        self.block04 = SENextBottleneck(encoder_channels[3], encoder_channels[4], stride=strides[3], is_shortcut=True, pool='avg')

        ### shared
        self.center = nn.Sequential(
          *[SENextBottleneck(encoder_channels[4], encoder_channels[4], stride=None, is_shortcut=False) for i in range(backbone[3])]
        )  

        ### MR decoder
        self.deblock04 = DecoderBlock(encoder_channels[-1], encoder_channels[-2], decoder_channels[0], strides[-1])
        self.deblock03 = DecoderBlock(decoder_channels[0], encoder_channels[-3], decoder_channels[1], strides[-2])
        self.deblock02 = DecoderBlock(decoder_channels[1], encoder_channels[-4], decoder_channels[2], strides[-3])
        self.deblock01 = DecoderBlock(decoder_channels[2], encoder_channels[-5], decoder_channels[3], strides[-4])

        ### MR seg head
        self.head0 = nn.Conv2d(decoder_channels[-1], 5, 1, 1, 0, bias=False)

        ### one more 
        backbone = [2,2,2,2,2]
        encoder_channels = [32, 64, 128, 192, 256, 512]
        decoder_channels = [256, 128, 64, 32, 16]
        strides = [[2,2], [2,2], [2,2], [2,2], [2,2]]

        ### CT encoder [512, 512]
        self.block10 = nn.Sequential(
            ConvBR2d(num_features, encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
        )
        self.block11 = nn.Sequential(
            SENextBottleneck(encoder_channels[0], encoder_channels[1], stride=strides[0], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[1], encoder_channels[1], stride=None, is_shortcut=False) for i in range(backbone[0])]
        )
        self.block12 = nn.Sequential(
            SENextBottleneck(encoder_channels[1], encoder_channels[2], stride=strides[1], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[2], encoder_channels[2], stride=None, is_shortcut=False) for i in range(backbone[1])]
        )
        self.block13 = nn.Sequential(
            SENextBottleneck(encoder_channels[2], encoder_channels[3], stride=strides[2], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[3], encoder_channels[3], stride=None, is_shortcut=False) for i in range(backbone[2])]
        )
        self.block14 = nn.Sequential(
            SENextBottleneck(encoder_channels[3], encoder_channels[4], stride=strides[3], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[4], encoder_channels[4], stride=None, is_shortcut=False) for i in range(backbone[3])]
        )
        self.block15 = SENextBottleneck(encoder_channels[4], encoder_channels[5], stride=strides[4], is_shortcut=True, pool='avg')
        

        ### CT decoder
        self.deblock15 = DecoderBlock(encoder_channels[-1], encoder_channels[-2], decoder_channels[0], strides[-1])
        self.deblock14 = DecoderBlock(decoder_channels[0], encoder_channels[-3], decoder_channels[1], strides[-2])
        self.deblock13 = DecoderBlock(decoder_channels[1], encoder_channels[-4], decoder_channels[2], strides[-3])
        self.deblock12 = DecoderBlock(decoder_channels[2], encoder_channels[-5], decoder_channels[3], strides[-4])
        self.deblock11 = DecoderBlock(decoder_channels[3], encoder_channels[-6], decoder_channels[4], strides[-5])


        ### CT seg head
        self.head1 = nn.Conv2d(decoder_channels[-1], 2, 1, 1, 0, bias=False)

        self.cnt = 0

    def forward(self, x, modality):
        if modality == 0:
            _x_ = x.detach().cpu().numpy() # [B, 1, 256, 256]

            x0 = self.block00(x)
            x1 = self.block01(x0)
            x2 = self.block02(x1)
            x3 = self.block03(x2)
            x4 = self.block04(x3)
            x = self.center(x4)
           
            _f_ = x.detach().cpu().numpy() # [B, 512, 32, 32]

            x = self.deblock04(x, x3)
            x = self.deblock03(x, x2)
            x = self.deblock02(x, x1)
            x = self.deblock01(x, x0)
            x = self.head0(x)

            _y_ = x.detach().cpu().numpy().argmax(1) # [B, 1, 256, 256]
            print(self.cnt, _x_.shape, _f_.shape, _y_.shape)
            torch.save({"x":_x_, "f":_f_, "y":_y_}, f'{modality}_{self.cnt}.bin')
            self.cnt += 1
            
        else:
            x0 = self.block10(x)
            x1 = self.block11(x0)
            x2 = self.block12(x1)
            x3 = self.block13(x2)
            x4 = self.block14(x3)
            x5 = self.block15(x4)
            x = self.center(x5)
            x = self.deblock15(x, x4)
            x = self.deblock14(x, x3)
            x = self.deblock13(x, x2)
            x = self.deblock12(x, x1)
            x = self.deblock11(x, x0)
            x = self.head1(x)            
        return x


# single trainer
# plan A: train MR fully, then train CT fully
def run_training_A(trainer, modality):
    torch.cuda.empty_cache()

    trainer._maybe_init_amp()

    maybe_mkdir_p(trainer.output_folder)

    if not trainer.was_initialized:
        trainer.initialize(True)

    while trainer.epoch < trainer.max_num_epochs:
        trainer.print_to_log_file("\nepoch: ", trainer.epoch)
        epoch_start_time = time()
        train_losses_epoch = []

        ####################
        #  train one epoch
        trainer.network.train()
        for b in range(trainer.num_batches_per_epoch):
            l = trainer.run_iteration(trainer.tr_gen, modality, True)
            train_losses_epoch.append(l)
        ####################

        trainer.all_tr_losses.append(np.mean(train_losses_epoch))
        trainer.print_to_log_file("train loss : %.4f" % trainer.all_tr_losses[-1])

        with torch.no_grad():
            # validation with train=False
            trainer.network.eval()
            val_losses = []

            ###############################
            # validate one epoch
            for b in range(trainer.num_val_batches_per_epoch):
                l = trainer.run_iteration(trainer.val_gen, modality, False, True)
                val_losses.append(l)
            ###############################

            trainer.all_val_losses.append(np.mean(val_losses))
            trainer.print_to_log_file("val loss (train=False): %.4f" % trainer.all_val_losses[-1])

        epoch_end_time = time()

        trainer.update_train_loss_MA()  # needed for lr scheduler and stopping of training

        continue_training = trainer.on_epoch_end()
        
        # logging to tensorboard
        trainer.writer.add_scalar(f"train/loss{modality}", trainer.all_tr_losses[-1], trainer.epoch)
        trainer.writer.add_scalar(f"evaulate/loss{modality}", trainer.all_val_losses[-1], trainer.epoch)
        trainer.writer.add_scalar(f"evaluate/metric{modality}", trainer.all_val_eval_metrics[-1], trainer.epoch)

        if not continue_training:
            # allows for early stopping
            break

        trainer.epoch += 1
        trainer.print_to_log_file("This epoch took %f s\n" % (epoch_end_time-epoch_start_time))

    trainer.save_checkpoint(join(trainer.output_folder, "model_final_checkpoint.model"))
    # now we can delete latest as it will be identical with final
    if isfile(join(trainer.output_folder, "model_latest.model")):
        os.remove(join(trainer.output_folder, "model_latest.model"))
    if isfile(join(trainer.output_folder, "model_latest.model.pkl")):
        os.remove(join(trainer.output_folder, "model_latest.model.pkl"))

    
# two trainers
# plan B: epoch-wise cross 
def run_training_B(trainer0, trainer1):
    torch.cuda.empty_cache()

    trainer0._maybe_init_amp()
    trainer1._maybe_init_amp()

    maybe_mkdir_p(trainer0.output_folder)
    maybe_mkdir_p(trainer1.output_folder)

    if not trainer0.was_initialized:
        trainer0.initialize(True)

    if not trainer1.was_initialized:
        trainer1.initialize(True)
    
    while trainer0.epoch < trainer0.max_num_epochs and trainer1.epoch < trainer1.max_num_epochs:
        
        epoch_start_time = time()

        ####################
        #  trainer0 one epoch
        trainer0.print_to_log_file("\n[Trainer0] epoch: ", trainer0.epoch)
        train_losses_epoch = []
        trainer0.network.train()
        for b in range(trainer0.num_batches_per_epoch):
            l = trainer0.run_iteration(trainer0.tr_gen, 0, True)
            train_losses_epoch.append(l)
        trainer0.all_tr_losses.append(np.mean(train_losses_epoch))
        trainer0.print_to_log_file("[Trainer0] train loss : %.4f" % trainer0.all_tr_losses[-1])
        ####################

        ####################
        #  trainer1 one epoch
        trainer1.print_to_log_file("\n[Trainer1] epoch: ", trainer1.epoch)
        train_losses_epoch = []
        trainer1.network.train()
        for b in range(trainer1.num_batches_per_epoch):
            l = trainer1.run_iteration(trainer1.tr_gen, 1, True)
            train_losses_epoch.append(l)
        trainer1.all_tr_losses.append(np.mean(train_losses_epoch))
        trainer1.print_to_log_file("[Trainer1] train loss : %.4f" % trainer1.all_tr_losses[-1])
        ####################


        with torch.no_grad():

            ###############################
            # validate0 one epoch
            trainer0.network.eval()
            val_losses = []
            for b in range(trainer0.num_val_batches_per_epoch):
                l = trainer0.run_iteration(trainer0.val_gen, 0, False, True)
                val_losses.append(l)
            trainer0.all_val_losses.append(np.mean(val_losses))
            trainer0.print_to_log_file("[Trainer0] val loss: %.4f" % trainer0.all_val_losses[-1])
            ###############################

            ###############################
            # validate1 one epoch
            trainer1.network.eval()
            val_losses = []
            for b in range(trainer1.num_val_batches_per_epoch):
                l = trainer1.run_iteration(trainer1.val_gen, 1, False, True)
                val_losses.append(l)
            trainer1.all_val_losses.append(np.mean(val_losses))
            trainer1.print_to_log_file("[Trainer1] val loss: %.4f" % trainer1.all_val_losses[-1])
            ###############################

        epoch_end_time = time()

        trainer0.update_train_loss_MA()  # needed for lr scheduler and stopping of training
        trainer1.update_train_loss_MA()  # needed for lr scheduler and stopping of training

        # this may cause saving two same models, but anyway it should work.
        continue_training = trainer0.on_epoch_end()
        continue_training = trainer1.on_epoch_end()

        # logging to tensorboard
        trainer0.writer.add_scalar("train/loss0", trainer0.all_tr_losses[-1], trainer0.epoch)
        trainer0.writer.add_scalar("evaulate/loss0", trainer0.all_val_losses[-1], trainer0.epoch)
        trainer0.writer.add_scalar("evaluate/metric0", trainer0.all_val_eval_metrics[-1], trainer0.epoch)
        # logging to tensorboard
        trainer1.writer.add_scalar("train/loss1", trainer1.all_tr_losses[-1], trainer1.epoch)
        trainer1.writer.add_scalar("evaulate/loss1", trainer1.all_val_losses[-1], trainer1.epoch)
        trainer1.writer.add_scalar("evaluate/metric1", trainer1.all_val_eval_metrics[-1], trainer1.epoch)

        trainer0.epoch += 1
        trainer1.epoch += 1

        # also, those two log files are differently placed.
        trainer0.print_to_log_file("This epoch took %f s\n" % (epoch_end_time-epoch_start_time))
        trainer1.print_to_log_file("This epoch took %f s\n" % (epoch_end_time-epoch_start_time))

    # save once is OK
    trainer0.save_checkpoint(join(trainer0.output_folder, "model_final_checkpoint.model"))
    # now we can delete latest as it will be identical with final
    if isfile(join(trainer0.output_folder, "model_latest.model")):
        os.remove(join(trainer0.output_folder, "model_latest.model"))
    if isfile(join(trainer0.output_folder, "model_latest.model.pkl")):
        os.remove(join(trainer0.output_folder, "model_latest.model.pkl"))


# two trainers
# plan C: step-wise cross 
def run_training_C(trainer0, trainer1):
    torch.cuda.empty_cache()

    trainer0._maybe_init_amp()
    trainer1._maybe_init_amp()

    maybe_mkdir_p(trainer0.output_folder)
    maybe_mkdir_p(trainer1.output_folder)

    if not trainer0.was_initialized:
        trainer0.initialize(True)

    if not trainer1.was_initialized:
        trainer1.initialize(True)
    
    while trainer0.epoch < trainer0.max_num_epochs and trainer1.epoch < trainer1.max_num_epochs:

        epoch_start_time = time()

        ####################
        #  trainer0 & trainer1 one epoch
        trainer0.print_to_log_file("\n[Trainer0] epoch: ", trainer0.epoch)
        trainer1.print_to_log_file("\n[Trainer1] epoch: ", trainer1.epoch)
        
        train_losses_epoch0 = []
        train_losses_epoch1 = []

        trainer0.network.train()
        trainer1.network.train()

        for b in range(trainer0.num_batches_per_epoch*2): # double steps
            if b%2 == 0:
                # trainer0
                l = trainer0.run_iteration(trainer0.tr_gen, 0, True)
                train_losses_epoch0.append(l)
            else:
                # trainer1
                l = trainer1.run_iteration(trainer1.tr_gen, 1, True)
                train_losses_epoch1.append(l)

        trainer0.all_tr_losses.append(np.mean(train_losses_epoch0))
        trainer1.all_tr_losses.append(np.mean(train_losses_epoch1))
        trainer0.print_to_log_file("[Trainer0] train loss : %.4f" % trainer0.all_tr_losses[-1])
        trainer1.print_to_log_file("[Trainer1] train loss : %.4f" % trainer1.all_tr_losses[-1])
        ####################


        with torch.no_grad():

            ###############################
            # validate0 one epoch
            trainer0.network.eval()
            val_losses = []
            for b in range(trainer0.num_val_batches_per_epoch):
                l = trainer0.run_iteration(trainer0.val_gen, 0, False, True)
                val_losses.append(l)
            trainer0.all_val_losses.append(np.mean(val_losses))
            trainer0.print_to_log_file("[Trainer0] val loss: %.4f" % trainer0.all_val_losses[-1])
            ###############################

            ###############################
            # validate1 one epoch
            trainer1.network.eval()
            val_losses = []
            for b in range(trainer1.num_val_batches_per_epoch):
                l = trainer1.run_iteration(trainer1.val_gen, 1, False, True)
                val_losses.append(l)
            trainer1.all_val_losses.append(np.mean(val_losses))
            trainer1.print_to_log_file("[Trainer1] val loss: %.4f" % trainer1.all_val_losses[-1])
            ###############################

        epoch_end_time = time()

        trainer0.update_train_loss_MA()  # needed for lr scheduler and stopping of training
        trainer1.update_train_loss_MA()  # needed for lr scheduler and stopping of training

        # this may cause saving two same models, but anyway it should work.
        continue_training = trainer0.on_epoch_end()
        continue_training = trainer1.on_epoch_end()

        # logging to tensorboard
        trainer0.writer.add_scalar("train/loss0", trainer0.all_tr_losses[-1], trainer0.epoch)
        trainer0.writer.add_scalar("evaulate/loss0", trainer0.all_val_losses[-1], trainer0.epoch)
        trainer0.writer.add_scalar("evaluate/metric0", trainer0.all_val_eval_metrics[-1], trainer0.epoch)
        # logging to tensorboard
        trainer1.writer.add_scalar("train/loss1", trainer1.all_tr_losses[-1], trainer1.epoch)
        trainer1.writer.add_scalar("evaulate/loss1", trainer1.all_val_losses[-1], trainer1.epoch)
        trainer1.writer.add_scalar("evaluate/metric1", trainer1.all_val_eval_metrics[-1], trainer1.epoch)

        trainer0.epoch += 1
        trainer1.epoch += 1

        # also, those two log files are differently placed.
        trainer0.print_to_log_file("This epoch took %f s\n" % (epoch_end_time-epoch_start_time))
        trainer1.print_to_log_file("This epoch took %f s\n" % (epoch_end_time-epoch_start_time))


    trainer0.save_checkpoint(join(trainer0.output_folder, "model_final_checkpoint.model"))
    # now we can delete latest as it will be identical with final
    if isfile(join(trainer0.output_folder, "model_latest.model")):
        os.remove(join(trainer0.output_folder, "model_latest.model"))
    if isfile(join(trainer0.output_folder, "model_latest.model.pkl")):
        os.remove(join(trainer0.output_folder, "model_latest.model.pkl"))

    trainer1.save_checkpoint(join(trainer1.output_folder, "model_final_checkpoint.model"))
    # now we can delete latest as it will be identical with final
    if isfile(join(trainer1.output_folder, "model_latest.model")):
        os.remove(join(trainer1.output_folder, "model_latest.model"))
    if isfile(join(trainer1.output_folder, "model_latest.model.pkl")):
        os.remove(join(trainer1.output_folder, "model_latest.model.pkl"))