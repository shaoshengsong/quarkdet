# QuarkDet implementation lightweight object detection based on PyTorch 

[中文(有更多的详细资料)](./README_cn.md)  [English](./README.md)

Here we implement lightweight object detection<br>
backbone support mobilenetv3、shufflenetv2、ghostnet、efficientnet<br>
neck support FPN（cnn）,PAN（cnn）、FPN_Slim（non-cnn），PAN_Slim（non-cnn）、BiFPN<br>
head support gfl（Generalized Focal Loss）、gfl v2(custom)<br>

# Test Environment

Ubuntu18.04<br>
PyTorch 1.7<br>
Python 3.6<br>



Different devices have different performance on the mobile side, and models with different capabilities can be run according to different performance. The library provides models for low-power mobile devices and powerful mobile devices.<br>
If the model runs on a low-power mobile device, backbone can choose shufflenetv2 or moiblenetv3, and neck choosing FAN_Slim, is equivalent to an implementation of nanodet.<br>
If it is a mobile device with strong performance, backbone can choose to only write b2 and b3 types in the effcientnet, configuration, and neck can choose BiFPN, to be the equivalent of an effcientdet implementation.<br>



Backbone support mobilenetv3、shufflenetv2、ghostnet、efficientnet<br>
neck support FPN（Convolution）,PAN（Convolution）、FPN_Slim（Non-convolution），PAN_Slim（Non-convolution）、BiFPN<br>
head support gfl（Generalized Focal Loss）、gfl v2(Custom version)<br>

quarkdet can use the following permutations and combinations<br>
EfficientNet + BiFPN + GFL<br>
GhostNet + PAN + GFL<br>
GhostNet + BiFPN + GFL<br>
MobileNetV3 + PAN/PAN_Slim + GFLv2<br>
ShuffleNetV2 + PAN/PAN_Slim + GFL and so on.<br>
Just change the command line to a different configuration file during training， config file in the quarkdet/config folder<br>

EfficientDet the original implementation is<br>
EfficientNet + BiFPN + Box/Class Head<br>
This place has been changed<br>
EfficientNet + BiFPN + GFL(Generalized Focal Loss)<br>

## support mosaic Data Augmentation

```
load_mosaic=False,
mosaic_probability=0.2
mosaic_area=16,
```



Quarkdet support to configure in the file, the sample file config/ghostnet_slim640.yml and the original mosaic data enhancement is different, not randomly change the size of the picture, but 4 pictures of the same size, using a fixed center, that is, 4 pictures, equal size, support 320416640 equal width and height of the same size.<br>
Load_mosaic: indicates whether to start data enhancement.<br>
What percentage of mosaic_probability: data is enhanced by mosaic data<br>
If the mosaic_area:GT bbox size is less than this threshold, it will be filtered out.<br>


## gfl v2 version

Slightly different from the implementation of the official website, it can rise a little, only after the decimal point.<br>
From class QuarkDetHead (GFLHead): # you can directly replace GFLHead with GFLHeadV2 <br>
For the original GFocalV2 implementation, please [reference](https://github.com/implus/GFocalV2)。<br>

# GhostNet is divided into full version and simplified version.

config file 
ghostnet_full.yml full network version
ghostnet_slim.yml simplified network version

GhostNet  following streamlines have been made<br>
Remove all the layers in the stage5 whose expansion size is equal to 960. the removed layers also include <br>.
Conv2d 1 × 1 the number of output channels equals 960 and 1280 layers, average pooling layer and last fully connected layer <br>

## mobilenetv3small version

The network is intercepted from the beginning to hs2 (bn2)<br>

## Train method

For Single-GPU config<br>
quarkdet.yml config example<br>
device:<br>
&emsp; gpu_ids: [0]<br>

### For Single-GPU run

```
python tools/train.py config/quarkdet.yml
```

For Multi-GPU config<br>
quarkdet.yml config example<br>
device:<br>
&emsp; gpu_ids: [0,1]<br>

### For Multi-GPU run

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port 30001 tools/train.py config/quarkdet.yml
```

## Inference video

It can be used for demonstration<br>

```
python ./demo/demo.py  'video' --path /media/ubuntu/data/1.mp4 --config config/efficientdet.yml --model ./workspace/efficientdet/model_best/model_best.pth
```

## Learning rate support ReduceLROnPlateau

When the continuous n times of the monitored index has not been improved, the learning rate is reduced, where n is patience in the configuration.

```
  lr_schedule:
    name: ReduceLROnPlateau
    mode: min
    factor: 0.1
    patience: 10
    verbose: True
    threshold: 0.00001
    threshold_mode: rel
    cooldown: 0
    min_lr: 0
    eps: 0.000000001 #1e-08
```

## ghostnet_full result

Resolution=320 * 320
epoch = 85

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.220
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.369
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.220
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.069
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.219
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.366
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.219
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.347
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.118
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.414
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.591
```

## ghostnet_slim result

Computational complexity:       0.56 GFLOPs<br>
Number of parameters:           1.77 M  <br>

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.198
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.339
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.198
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.059
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.197
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.323
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.211
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.340
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.105
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.410
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.583
```

If it is distributed training, it can be done in norm_cfg.
Dict (type='BN', momentum=0.01,eps=1e-3, requires_grad=True).
Type='BN' changed to type='SyncBN'
Because no judgment is made here whether it is distributed or not, BN is written in between.

## efficientdet

EfficientNet + BiFPN + GFL<br>
The original was a feature of 5 level, which was reduced to 3 level here.<br>
Automatic learning rate，epoch=190<br>

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.230
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.369
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.237
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.078
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.246
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.357
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.236
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.377
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.397
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.459
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.609
```

Download<br> 
[efficientdet-b2 download link](https://pan.baidu.com/s/1-_G5wWQwCPeHaahXbfarBQ) <br> 
code：hl3o <br>


## Muchas gracias.

https://github.com/huawei-noah/ghostnet<br>
https://github.com/xiaolai-sqlai/mobilenetv3<br>
https://github.com/RangiLyu/nanodet<br>
https://github.com/ultralytics/yolov5<br>
https://github.com/implus/GFocal<br>
https://github.com/implus/GFocalV2<br>