import copy
from .fpn import FPN
from .pan import PAN
from .bifpn import BiFPN
from .fpn_slim import FPN_Slim
from .pan_slim import PAN_Slim


def build_fpn(cfg):
    fpn_cfg = copy.deepcopy(cfg)
    name = fpn_cfg.pop('name')
    if name == 'FPN':
        return FPN(**fpn_cfg)
    elif name == 'PAN':
        return PAN(**fpn_cfg)
    elif name == 'BiFPN':
        return BiFPN(**fpn_cfg)
    elif name == 'FPN_Slim':
            return FPN_Slim(**fpn_cfg)
    elif name == 'PAN_Slim':
        return PAN_Slim(**fpn_cfg)
    else:
        raise NotImplementedError