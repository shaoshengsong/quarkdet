import torch.nn as nn
import torch.nn.functional as F
from ..module.conv import ConvModule
from .fpn import FPN
import numpy as np


class PAN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(PAN,
              self).__init__(in_channels, out_channels, num_outs, start_level,
                             end_level, conv_cfg, norm_cfg, activation)
        #显示调用基类的__init__方法，Python不会自动执行这些初始化操作。     
        print("PAN:",in_channels, out_channels, num_outs, start_level,end_level, conv_cfg, norm_cfg, activation)
        self.init_weights()
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pan_convs = nn.ModuleList()
        
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=None,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=None,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.pan_convs.append(pafpn_conv)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        #print("PAN forward:",self.in_channels, self.out_channels, self.num_outs, self.start_level,self.end_level)


        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals) 
        for i in range(used_backbone_levels - 1, 0, -1):   #i=[2,1]
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear',align_corners=True)

        # part 1: from original levels

        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]


        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pan_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])


      
     
        return tuple(outs)
