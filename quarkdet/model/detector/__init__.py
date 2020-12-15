from .gfl import GFL


def build_model(model_cfg):
    if model_cfg.detector.name == 'GFL':
        model = GFL(model_cfg.detector.backbone, model_cfg.detector.neck, model_cfg.detector.head)
    else:
        raise NotImplementedError
    return model
