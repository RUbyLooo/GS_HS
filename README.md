# GS_HS
本项目的目标是探索如何在仿真环境中可靠地评估真实世界的机器人操作策略
# installation

```conda create -n env python=3.10.9 
conda activate env 
pip install -r requirements.txt

cd pyroboplan
pip install -e .
pip install numpy==1.24.0 
pip install beartype
pip install wandb
pip install tensorboard
pip install ema_pytorch
pip install omegaconf```
