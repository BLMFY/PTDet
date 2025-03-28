import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['merge_cat', 'merge_add', 'merge_mul']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.ReLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))



class merge_cat(nn.Module):
    
    def __init__(self, in_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.up_conv = nn.ConvTranspose2d(1, 1, 4, 4)
        self.con_conv = Conv(in_channels*2, in_channels, 3, 1) 

        self.f_conv = Conv(in_channels, in_channels, 3, 1) 

    def forward(self, fuse, gauss_map):
        gauss_map = gauss_map.float()
        # gauss_map_out = F.interpolate(gauss_map, scale_factor=16, mode='bilinear')
        gauss_map = self.up_conv(gauss_map)

        fuse_map_with_gauss = self.con_conv(torch.cat([gauss_map*fuse + gauss_map, fuse],dim=1))
        
        fuse = self.f_conv(fuse_map_with_gauss)
        return fuse

