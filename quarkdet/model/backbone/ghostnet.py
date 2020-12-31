# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..module.activation import act_layers


def get_url(width_mult=1.0):
    if width_mult==1.0:
        return 'https://github.com/huawei-noah/ghostnet/raw/master/pytorch/models/state_dict_93.98.pth'
    else:
        logging.info('GhostNet only has 1.0 pretrain model. ')
        return None


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

#---------------------------------------------------------------------

def get_ld_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)

def get_dct_weights(width, height, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, height)

    # split channel for multi-spectral attention
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                val = get_ld_dct(t_x, u_x, width) * get_ld_dct(t_y, v_y, height)
                dct_weights[:, i * c_part: (i+1) * c_part, t_x, t_y] = val

    return dct_weights
#---------------------------------------------------------------------

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act="ReLU", gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()

        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layers(act)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
  
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act="ReLU"):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layers(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, act="ReLU"):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            act_layers(act) if act else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            act_layers(act) if act else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act="ReLU", se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, act=act)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, act=None)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet_slim(nn.Module):
    def __init__(self, width_mult=1.0, out_stages=(4, 6, 9), act='ReLU', pretrain=True):
        super(GhostNet_slim, self).__init__()
        self.width_mult = width_mult
        self.out_stages = out_stages
        # setting of inverted residual blocks
        self.cfgs = [
            # k, t,   c,  SE, s
            # stage1
            [[3, 16,  16, 0, 1]],     # 0
            # stage2
            [[3, 48,  24, 0, 2]],     # 1
            [[3, 72,  24, 0, 1]],     # 2  1/4
            # stage3
            [[5, 72,  40, 0.25, 2]],  # 3
            [[5, 120, 40, 0.25, 1]],  # 4  1/8
            # stage4
            [[3, 240, 80, 0, 2]],     # 5
            [[3, 200, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]
             ],                       # 6  1/16
            # stage5
            [[5, 672, 160, 0.25, 2]], # 7
            # [[5, 960, 160, 0, 1],
            #  [5, 960, 160, 0.25, 1],
            #  [5, 960, 160, 0, 1],
            #  [5, 960, 160, 0.25, 1]
            #  ]                        # 8
            
            
            
        #    # k, t, c, SE, s
        #     [3, 16, 16, 0, 1],
        #     [3, 48, 24, 0, 2],
        #     [3, 72, 24, 0, 1],
        #     [5, 72, 40, 1, 2],
        #     [5, 120, 40, 1, 1],
        #     [3, 240, 80, 0, 2],
        #     [3, 200, 80, 0, 1],
        #     [3, 184, 80, 0, 1],
        #     [3, 184, 80, 0, 1],
        #     [3, 480, 112, 1, 1],
        #     [3, 672, 112, 1, 1],
        #     [5, 672, 160, 1, 2],
        #     [5, 960, 160, 0, 1],
        #     [5, 960, 160, 1, 1],
        #     [5, 960, 160, 0, 1],
        #     [5, 960, 160, 1, 1]
        ]
        #  ------conv+bn+act----------# 9  1/32

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = act_layers(act)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width_mult, 4)
                hidden_channel = _make_divisible(exp_size * width_mult, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    act=act, se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width_mult, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1, act=act)))  #9

        self.blocks = nn.Sequential(*stages)

        self._initialize_weights(pretrain)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        output = []
        #for i in range(10):
        for i in range(8):
            x = self.blocks[i](x)
            if i in self.out_stages:
                output.append(x)
            #print("ghost output:",x.shape)
            #print("----------------------------------------------------")
                # ghost output: torch.Size([80, 40, 40, 40])
                # ghost output: torch.Size([80, 112, 20, 20])
                # ghost output: torch.Size([80, 960, 10, 10])
                
# |    |    └─Sequential: 3-1                        [-1, 16, 160, 160]        464
# |    |    └─Sequential: 3-2                        [-1, 24, 80, 80]          2,564
# |    |    └─Sequential: 3-3                        [-1, 24, 80, 80]          2,352
# |    |    └─Sequential: 3-4                        [-1, 40, 40, 40]          9,636
# |    |    └─Sequential: 3-5                        [-1, 40, 40, 40]          13,672
# |    |    └─Sequential: 3-6                        [-1, 80, 20, 20]          22,920
# |    |    └─Sequential: 3-7                        [-1, 112, 20, 20]         533,476
# |    |    └─Sequential: 3-8                        [-1, 160, 10, 10]         362,840
# |    |    └─Sequential: 3-9                        [-1, 160, 10, 10]         1,567,520
# |    |    └─Sequential: 3-10                       [-1, 960, 10, 10]         155,520



        return tuple(output)

    def _initialize_weights(self, pretrain=True):
        print('init weights...')
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv_stem' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if pretrain:
            url = get_url(self.width_mult)
            if url is not None:
                state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
                self.load_state_dict(state_dict, strict=False)


class GhostNet_full(nn.Module):
    def __init__(self, width_mult=1.0, out_stages=(4, 6, 9), act='ReLU', pretrain=True):
        super(GhostNet_full, self).__init__()
        self.width_mult = width_mult
        self.out_stages = out_stages
        # setting of inverted residual blocks
        self.cfgs = [
            # k, t,   c,  SE, s
            # stage1
            [[3, 16,  16, 0, 1]],     # 0
            # stage2
            [[3, 48,  24, 0, 2]],     # 1
            [[3, 72,  24, 0, 1]],     # 2  1/4
            # stage3
            [[5, 72,  40, 0.25, 2]],  # 3
            [[5, 120, 40, 0.25, 1]],  # 4  1/8
            # stage4
            [[3, 240, 80, 0, 2]],     # 5
            [[3, 200, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]
             ],                       # 6  1/16
            # stage5
            [[5, 672, 160, 0.25, 2]], # 7
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]
             ]                        # 8
            
        ]
        #  ------conv+bn+act----------# 9  1/32

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = act_layers(act)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width_mult, 4)
                hidden_channel = _make_divisible(exp_size * width_mult, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    act=act, se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width_mult, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1, act=act)))  #9

        self.blocks = nn.Sequential(*stages)

        self._initialize_weights(pretrain)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        output = []
        for i in range(10):
            x = self.blocks[i](x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _initialize_weights(self, pretrain=True):
        print('init weights...')
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv_stem' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if pretrain:
            url = get_url(self.width_mult)
            if url is not None:
                state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
                self.load_state_dict(state_dict, strict=False)