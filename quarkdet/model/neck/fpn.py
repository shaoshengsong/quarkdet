

import torch.nn as nn
import torch.nn.functional as F
from ..module.conv import ConvModule
from ..module.init_weights import xavier_init


class FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None
                 ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # for i in range(self.start_level, self.backbone_end_level):
        #     l_conv = ConvModule(
        #         in_channels[i],
        #         out_channels,
        #         1,
        #         conv_cfg=conv_cfg,
        #         norm_cfg=norm_cfg,
        #         activation=activation,
        #         inplace=False)

        #     self.lateral_convs.append(l_conv)
        
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                #act_cfg=act_cfg,
                activation=activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)    
        print("FPN:",self.lateral_convs)    
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear',align_corners=True)

        # build outputs
        # outs = [
        #     # self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        #     laterals[i] for i in range(used_backbone_levels)
        # ]
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        #这里去除了fpn_convs stride=(2, 2)的两层，lateral_convs和fpn_convs都是三层
        return tuple(outs)


# if __name__ == '__main__':
