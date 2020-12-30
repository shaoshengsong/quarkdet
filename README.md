# quarkdet 
weightlight object detection<br>
GhostNet + PAN + GFL<br>

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

ghostnet精简版本<br>
对GhostNet做了以下精简<br>
取出stage5中expansion size等于960的所有层，去除的层还包括<br>
Conv2d 1×1 the number of output channels等于960和1280的层，平均池化层和最后的全连接层<br>

mobilenetv3small版本<br>
网络从开头截取到hs2(bn2)<br>

此repo代码中GFocalV2未完全更改完成，如要实现请[参考](https://github.com/implus/GFocalV2)。

## Muchas gracias.

https://github.com/huawei-noah/ghostnet

https://github.com/xiaolai-sqlai/mobilenetv3

https://github.com/RangiLyu/nanodet  (特别感谢)

https://github.com/ultralytics/yolov5

https://github.com/implus/GFocal

https://github.com/implus/GFocalV2