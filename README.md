# quarkdet 
lightweight object detection<br>
轻量级目标检测<br>
GhostNet + PAN + GFL<br>

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
backbone 支持 mobilenetv3, shufflenetv2, ghostnet<br>
neck 支持 FPN,PAN（卷积）<br>
head 支持 gfl<br>

## Muchas gracias.

https://github.com/huawei-noah/ghostnet

https://github.com/xiaolai-sqlai/mobilenetv3

https://github.com/RangiLyu/nanodet  (特别感谢)

https://github.com/ultralytics/yolov5

https://github.com/implus/GFocal

https://github.com/implus/GFocalV2