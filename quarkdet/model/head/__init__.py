import copy
from .gfl_head import GFLHead
from .quarkdet_head import QuarkDetHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'GFLHead':
        return GFLHead(**head_cfg)
    elif name == 'QuarkDetHead':
        return QuarkDetHead(**head_cfg)
    else:
        raise NotImplementedError