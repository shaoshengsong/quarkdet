import copy
from .gfl_headv2 import GFLHeadV2
from .quarkdet_head import QuarkDetHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'GFLHeadV2':
        return GFLHeadV2(**head_cfg)
    elif name == 'QuarkDetHead':
        return QuarkDetHead(**head_cfg)
    else:
        raise NotImplementedError