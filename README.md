# PyTorch实现 轻量级目标检测 quarkdet 
Here we implement lightweight object detection<br>
backbone support mobilenetv3、shufflenetv2、ghostnet、efficientnet<br>
neck support FPN（cnn）,PAN（cnn）、FPN_Slim（non-cnn），PAN_Slim（non-cnn）、BiFPN<br>
head support gfl（Generalized Focal Loss）、gfl v2(custom)<br>


在移动端不同的设备有不同的性能，针对不同的性能可以运行不同能力的模型.该库提供了低功耗的移动端设备和性能强劲的移动设备所使用的模型。<br>
如果模型是运行在低功耗的移动端设备，backbone可以选择shufflenetv2或者moiblenetv3等，neck选择FAN_Slim，相当于一个nanodet的实现。<br>
如果是性能强劲的移动设备，backbone可以选择effcientnet,配置中只写了b2和b3类型，neck可以选择BiFPN，相当于一个effcientdet的实现。<br>

backbone 支持 mobilenetv3、shufflenetv2、ghostnet、efficientnet<br>
neck 支持 FPN（卷积）,PAN（卷积）、FPN_Slim（非卷积），PAN_Slim（非卷积）、BiFPN<br>
head 支持 gfl（Generalized Focal Loss）、gfl v2(自定义版本)<br>

quarkdet可以用使用以下排列组合方式<br>
EfficientNet + BiFPN + GFL<br>
GhostNet + PAN + GFL<br>
GhostNet + BiFPN + GFL<br>
MobileNetV3 + PAN/PAN_Slim + GFLv2<br>
ShuffleNetV2 + PAN/PAN_Slim + GFL等等可以随意组合<br>
训练时将命令行换成不同的配置文件即可，配置文件在quarkdet/config文件夹中<br>

关于EfficientDet原版实现是<br>
EfficientNet + BiFPN + Box/Class Head<br>
这里改造为<br>
EfficientNet + BiFPN + GFL(Generalized Focal Loss)<br>

## 支持mosaic数据增强
```
load_mosaic=False,
mosaic_probability=0.2
mosaic_area=16,
```
与原版mosaic数据增强不同，不是随机改变图片大小而是4张图片大小相同，<br>
load_mosaic：表示是否启动数据增强<br>
mosaic_probability：有多少比例的数据采用mosaic数据增强<br>
mosaic_area：GT bbox大小小于该阈值则过滤掉<br>

## gfl v2版本
与官网实现略有不同，能涨点，只是小数点之后。<br>
可以从class QuarkDetHead(GFLHead): # 可以直接将GFLHead替换成 GFLHeadV2<br>
如要原版GFocalV2实现请[参考](https://github.com/implus/GFocalV2)。<br>

# GhostNet分为完整版和精简版
配置文件 
ghostnet_full.yml 完整版
ghostnet_slim.yml 精简版

对GhostNet做了以下精简<br>
取出stage5中expansion size等于960的所有层，去除的层还包括<br>
Conv2d 1×1 the number of output channels等于960和1280的层，平均池化层和最后的全连接层<br>

## mobilenetv3small版本
网络从开头截取到hs2(bn2)<br>

## 使用方法
Single-GPU<br>
quarkdet.yml config example<br>
device:<br>
&emsp; gpu_ids: [0]<br>
```
python tools/train.py config/quarkdet.yml
```
Multi-GPU<br>
quarkdet.yml config example<br>
device:<br>
&emsp; gpu_ids: [0,1]<br>

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port 30001 tools/train.py config/quarkdet.yml
```


## 学习率支持 ReduceLROnPlateau
当监控的指标连续n次数还没有改进时,降低学习率，这里的n在配置里是patience
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
## ghostnet_full版本训练结果

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

## ghostnet_slim版本训练结果
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
如果是分布式训练，可以在norm_cfg中<br>
dict(type='BN', momentum=0.01,eps=1e-3, requires_grad=True)<br>
type='BN'更改为 type='SyncBN'<br>
因为这里没有做判断是否是分布式，所以就之间写了BN<br>
## efficientdet
EfficientNet + BiFPN + GFL<br>
原来的是5个level的特征，这里减少至3个level。<br>
自动学习率，epoch=190<br>
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
[链接](https://pan.baidu.com/s/1-_G5wWQwCPeHaahXbfarBQ) <br> 
提取码：hl3o <br>


## 关于技术
[目标检测 - Generalized Focal Loss 综述](https://blog.csdn.net/flyfish1986/article/details/110143467)<br>
[目标检测 - IoU和GIoU作为边框回归的损失和代码实现](https://blog.csdn.net/flyfish1986/article/details/110005818)<br>
[目标检测 - Neck的设计 PAN（Path Aggregation Network）](https://blog.csdn.net/flyfish1986/article/details/110520667)<br>
[目标检测 - Generalized Focal Loss的Anchor处理机制](https://blog.csdn.net/flyfish1986/article/details/110245329)<br>
其他可参考<br>
[目标检测 FCOS(FCOS: Fully Convolutional One-Stage Object Detection)](https://blog.csdn.net/flyfish1986/article/details/109809571)<br>
[目标检测 PAA 概率anchor分配算法（Probabilistic Anchor Assignment Algorithm）](https://blog.csdn.net/flyfish1986/article/details/109680310)<br>
[目标检测 PAA - 高斯混合模型（GMM）和期望最大化算法（EM algorithm）](https://blog.csdn.net/flyfish1986/article/details/109629048)<br>


## Muchas gracias.

https://github.com/huawei-noah/ghostnet<br>
https://github.com/xiaolai-sqlai/mobilenetv3<br>
https://github.com/RangiLyu/nanodet  (特别感谢)<br>
https://github.com/ultralytics/yolov5<br>
https://github.com/implus/GFocal<br>
https://github.com/implus/GFocalV2<br>