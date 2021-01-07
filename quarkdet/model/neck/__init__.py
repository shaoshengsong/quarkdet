import copy
from .fpn import FPN
from .pan import PAN
from .bifpn import BiFPN


def build_fpn(cfg):
    fpn_cfg = copy.deepcopy(cfg)
    name = fpn_cfg.pop('name')
    if name == 'FPN':
        return FPN(**fpn_cfg)
    elif name == 'PAN':
        return PAN(**fpn_cfg)
    elif name == 'BiFPN':
        return BiFPN(**fpn_cfg)
    else:
        raise NotImplementedError