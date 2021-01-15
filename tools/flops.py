import torch
import sys  
sys.path.append("./") 
from quarkdet.model.detector import build_model
from quarkdet.util import cfg, load_config, get_model_complexity_info


def main(config, input_shape=(3, 320, 320)):
    model = build_model(config.model)
    #flops, params = get_model_complexity_info(model, input_shape)

    macs, params = get_model_complexity_info(model, input_shape, as_strings=True,
                                           print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    cfg_path = r"config/ghostnet_slim.yml"
    load_config(cfg, cfg_path)
    main(config=cfg,
         input_shape=(3, 320, 320)
         )


