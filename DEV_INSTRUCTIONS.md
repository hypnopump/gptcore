# Development Instructions

## Env setup

```bash
conda create -n gptcore python=3.11 --yes 
conda activate gptcore
pip3 install torch torchvision torchaudio 
pip install -r requirements.txt
pip install lightning deepspeed einops transformers datasets wandb torchdata
```
* WARNING! Seems not to work with class inheritance?

## Docker: 

```bash
sudo docker run -td --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia --gpus all -it -v /home/eric:/pc pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel 
sudo docker ps
sudo nvidia-smi -lgc 210,1500

export CONTAINER_NAME="beautiful_snyder"
sudo docker exec -it $CONTAINER_NAME /bin/bash

# sudo docker kill $CONTAINER_NAME 


cd /pc
cd Desktop/projects/gptcore_private/

pip install -r requirements.txt
pip install lightning deepspeed einops transformers datasets wandb torchdata schedulefree
pip install triton==2.2.0 matplotlib
# commit: 904708643a7d018a9cc71c60159ed4bc597ae307 # last tested
pip install -e flash-linear-attention/.

apt-get update
apt-get install -y tmux

wandb login

```
* WARNING! I had bad experiences with docker, runs being killed without warning (whole container killed)

## Train

```bash
wandb: ad5c659f21e8269b521573221a2ef3bba980afe6
```

```bash
timeout 6000s python cli.py train -c configs/experimental/gptab.cfg.py
timeout 6000s python cli.py train -c configs/experimental/gptab.cfg.py
ls
```


```bash
# repro based
python cli.py train -c configs/experimental/gptab_hypno.cfg.py

# repro rwkv6_0x
python cli.py train -c "configs/experimental/rwkv6_0x_(k=1-w).cfg.py"
python cli.py train -c "configs/experimental/rwkv6_0x.cfg.py"

# repro rwkv5
python cli.py train -c configs/experimental/rwkv5_2x.cfg.py


### Experiment on MLP - with based deterministic
python cli.py train -c configs/gptalpha.cfg.py

python cli.py train -c configs/experimental/gptab_mlp.cfg.py
python cli.py train -c configs/experimental/memtention_0x.cfg.py
```

## Experiment queue: 

```bash
### MLP improve
# RMSNormMLP
# gate MLP (no act -> if not -> tanh)

# for triton development
export TRITON_INTERPRET=0

```