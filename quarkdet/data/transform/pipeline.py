from .warp import warp_and_resize
from .color import color_aug_and_norm
import functools


class Pipeline:
    def __init__(self,
                 cfg,
                 keep_ratio):
        self.warp = functools.partial(warp_and_resize,
                                      warp_kwargs=cfg,
                                      keep_ratio=keep_ratio)
        self.color = functools.partial(color_aug_and_norm,
                                       kwargs=cfg)

    def __call__(self, meta, dst_shape):
        meta = self.warp(meta=meta, dst_shape=dst_shape)
        meta = self.color(meta=meta)
        return meta
    
    
    
    
#经过两个数据增强，重要的是参数是否配置，如果配置则启用，不配置不启用，程序中首先判断参数是否存在
# functools.partial(func, /, *args, **keywords)
# Return a new partial object which when called will behave like func called with the positional arguments args and keyword arguments keywords. 
# If more arguments are supplied to the call, they are appended to args. 
# If additional keyword arguments are supplied, they extend and override keywords. 