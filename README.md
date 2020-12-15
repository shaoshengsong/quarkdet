# quarkdet
GhostNet + PAN + GFL<br>

Single-GPU<br>
quarkdet.yml config example<br>
device:<br>
  gpu_ids: [0]<br>
python tools/train.py config/quarkdet.yml<br> 

Multi-GPU<br>
quarkdet.yml config example<br>
device:<br>
  gpu_ids: [0,1]<br>


python -m torch.distributed.launch --nproc_per_node=2 --master_port 30001 tools/train.py config/quarkdet.yml<br>
