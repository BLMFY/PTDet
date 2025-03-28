import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath,  trunc_normal_
import math
import time
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv_dif(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7, mode='hv', act = False):

        super(Conv_dif, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.mode = mode
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'silu':
            self.act = self.default_act
        if act == 'h_swish':
            self.act = h_swish()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return self.act(self.bn(out_normal)) 
        else:
            #pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            if self.mode == 'hv':
                dif_weight = self.conv.weight.view(C_out, C_in, -1)[:, :, [1, 3, 4, 5, 7]].sum(2)
                dif_weight = dif_weight[:, :, None, None]
                out_diff = F.conv2d(input=x, weight=dif_weight, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            elif self.mode == 'an':
                dif_weight = self.conv.weight.view(C_out, C_in, -1)[:, :, [0, 2, 4, 6, 8]].sum(2)
                dif_weight = dif_weight[:, :, None, None]
    
                out_diff = F.conv2d(input=x, weight=dif_weight, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            return self.act(self.bn(out_normal - self.theta * out_diff))

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act='silu'):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act == 'silu':
            self.act = self.default_act
        if act == 'h_swish':
            self.act = h_swish()
        else:
            self.act = nn.Identity()
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
 
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
 
    def forward(self, x):
        return self.relu(x + 3) / 6
 
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
 
    def forward(self, x):
        return x * self.sigmoid(x)
 
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        mip = max(8, inp // reduction)
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
 
    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return out
    
class Attention_agent_hv(nn.Module):
    def __init__(self, dim,
                 num_heads=8,
                 agent_num = 64,
                 direct = 'h'):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = dim // num_heads
        self.agent_num = agent_num
        self.scale = self.key_dim ** -0.5
        # nh_kd = nh_kd = self.key_dim * num_heads
        # h = dim + nh_kd * 2
        self.direct = direct
        self.qkv = Conv(dim, 3*dim, 1, act=False)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, dim//32)
        self.cv1 = Conv(dim, mip, 1, act='silu')
        self.cv2 = Conv(mip, dim, 1, act='silu')
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, H, W).split([self.key_dim*self.num_heads, self.key_dim*self.num_heads, self.key_dim*self.num_heads], dim=1)
        if self.direct == 'h':
            print(k.shape)
            a_k = self.pool_h(k)
            a_v = self.pool_h(v)
            k, v = self.cv2(self.cv1(a_k)) * k, self.cv2(self.cv1(a_v)) * v
        elif self.direct == 'w':
            a_k = self.pool_w(k)
            a_v = self.pool_w(v)
            k, v = self.cv2(self.cv1(a_k)) * k, self.cv2(self.cv1(a_v)) * v
        q = q.reshape(B, self.num_heads, self.key_dim, H*W)
        k = k.reshape(B, self.num_heads, self.key_dim, H*W)
        v = v.reshape(B, self.num_heads, self.key_dim, H*W)
        # channels = num_head * head_dim
        a = self.pool(q.reshape(B, self.key_dim * self.num_heads, H, W)).\
                        reshape(B, self.key_dim * self.num_heads, -1).permute(0, 2, 1)
        a = a.reshape(B, self.agent_num, self.num_heads, self.key_dim).permute(0, 2, 3, 1)

        # b, num_head, agent_num, hw
        # attn_k = self.softmax((a.transpose(-2, -1) @ k) * self. scale + position_bias)
        attn_k = self.softmax((a.transpose(-2, -1) @ k) * self. scale)

        # b, num_head, agent_num, dim
        attn = (attn_k @ v.transpose(-2, -1))

        q_attn = self.softmax((q.transpose(-2, -1) @ a) * self.scale)
        # b, num_head, dim, hw
        x = (q_attn @ attn).transpose(-2, -1)
        x = x.reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class Attention_agent_df(nn.Module):
    def __init__(self, dim,
                 num_heads=8,
                 agent_num = 64,
                 mode = 'hv'):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = dim // num_heads
        self.agent_num = agent_num
        self.scale = self.key_dim ** -0.5
        self.qkv = Conv_dif(dim, 3*dim, theta=0.7, mode=mode, act=False)
        # self.qkv_ = Conv(dim, 3*dim, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        B, C, H, W = x.shape
        N = H * W
        t0 = time.time()
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, H, W).split([self.key_dim*self.num_heads, self.key_dim*self.num_heads, self.key_dim*self.num_heads], dim=1)
        q = q.reshape(B, self.num_heads, self.key_dim, H*W)
        k = k.reshape(B, self.num_heads, self.key_dim, H*W)
        v = v.reshape(B, self.num_heads, self.key_dim, H*W)
        # channels = num_head * head_dim
        a = self.pool(q.reshape(B, self.key_dim * self.num_heads, H, W)).\
                        reshape(B, self.key_dim * self.num_heads, -1).permute(0, 2, 1)
        a = a.reshape(B, self.agent_num, self.num_heads, self.key_dim).permute(0, 2, 3, 1)

        attn_k = self.softmax((a.transpose(-2, -1) @ k) * self. scale)

        # b, num_head, agent_num, dim
        attn = (attn_k @ v.transpose(-2, -1))

        q_attn = self.softmax((q.transpose(-2, -1) @ a) * self.scale)
        # b, num_head, dim, hw
        x = (q_attn @ attn).transpose(-2, -1)
        x = x.reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)

        return x

class Attention_agent(nn.Module):
    def __init__(self, dim, window_size,
                 num_heads=8,
                 agent_num = 64):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = dim // num_heads
        self.agent_num = agent_num
        self.scale = self.key_dim ** -0.5
        # nh_kd = nh_kd = self.key_dim * num_heads
        # h = dim + nh_kd * 2
        h = 3* dim
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)
        self.window_size = window_size  # Wh, Ww

        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*3, N).split([self.key_dim, self.key_dim, self.key_dim], dim=2)
        # channels = num_head * head_dim
        a = self.pool(q.reshape(B, H, W, self.key_dim * self.num_heads).permute(0, 3, 1, 2)).\
                        reshape(B, self.key_dim * self.num_heads, -1).permute(0, 2, 1)
        a = a.reshape(B, self.agent_num, self.num_heads, self.key_dim).permute(0, 2, 3, 1)

        position_bias1 = nn.functional.interpolate(self.an_bias, size=self.window_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
        position_bias = position_bias1 + position_bias2

        # b, num_head, agent_num, hw
        attn_k = self.softmax((a.transpose(-2, -1) @ k) * self. scale + position_bias)

        # b, num_head, agent_num, dim
        attn = (attn_k @ v.transpose(-2, -1))

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(B, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2

        # b, num_head, hw, agent_num
        q_attn = self.softmax((q.transpose(-2, -1) @ a) * self.scale + agent_bias)

        # b, num_head, dim, hw
        x = (q_attn @ attn).transpose(-2, -1)
        x = x.reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        print(x.shape)
        return x

class PSA_hv(nn.Module):

    def __init__(self, c1, e=0.5):
        super().__init__()

        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1, 1)
        self.hor_attn = Attention_agent_hv(dim=self.c, num_heads=self.c // 64, direct= 'h')
        self.ver_attn = Attention_agent_hv(dim=self.c, num_heads=self.c // 64, direct= 'w')

        self.ffn = nn.Sequential(
            Conv(self.c*2, self.c*2, 1),
            Conv(self.c*2, self.c*2, 1, act=False)
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        a = a + self.hor_attn(a)
        b = b + self.ver_attn(b)
        c = torch.cat((a, b), 1)
        c = c + self.ffn(c)

        return self.cv2(c)

    
class PSA_dif(nn.Module):

    def __init__(self, c1=64, e=0.5): # 256
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c1*4, c1, 3, 2, 1),
                                  nn.Conv2d(c1, c1, 3, 2, 1),)
        
        self.conv_up = Conv(c1, c1*4, 1, 1)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1, 1)
        
        self.hv_attn = Attention_agent_df(dim=self.c, num_heads=self.c // 16, mode='hv') #64
        self.an_attn = Attention_agent_df(dim=self.c, num_heads=self.c // 16, mode='an') #64

        self.ffn = nn.Sequential(
            Conv(self.c*2, self.c*2, 1),
            Conv(self.c*2, self.c*2, 1, act=False)
        )
        self.binarize = nn.Sequential(
            nn.Conv2d(c1*4, c1 
                        , 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, c1, 2, 2),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, 1, 2, 2),
            nn.Sigmoid())
        
    def forward(self, fuse):  
        '''
        fuse: B, 256, 160, 160 
        '''  

        x = self.conv(fuse) # B, 256, 40, 40 
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.an_attn(b)
        a = a + self.hv_attn(a)

        c = torch.cat((a, b), 1)
        c = self.cv2(c + self.ffn(c))
        atten_out = F.interpolate(x+c , scale_factor=4, mode='nearest')
        atten_out = self.conv_up(atten_out)
        output = self.binarize(atten_out+fuse)
        return output

if __name__  == "__main__":
    import time

    psa_dif = PSA_dif().to('cuda:0')
    x = torch.ones(8, 256, 160, 160).to('cuda:0')
    t0 = time.time()
    for i in range(100):
        r = psa_dif(x)
    print(time.time() - t0)