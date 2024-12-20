#!/bin/bash

## This is not meant to be an executable. The .sh extension is for syntax highlighting.

docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(dirname `pwd`):/haha nvcr.io/nvidia/nemo:24.09 /bin/bash


## Within container

# Let's prepare a nemo checkpoint of a tiny, toy llama3 model. This will be the
# nemo checkpoint we're going to export to HF format.
#
# Add --yes to skip the prompt.
/haha/02-cpt/01-pretrain.sh log.ckpt.save_optim_on_train_end=False
# ...
# 52M     /tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last
# 52M     total

# And here we go...
# NOTE: all paths hardcoded in the .py file :)
python /haha/04-export-ckpt/ckpt-nemo2hf.py
# ...
# [NeMo I 2024-11-28 09:58:11 base:44] Padded vocab_size: 50304, original vocab_size: 50257, dummy tokens: 47.
# CPU RNG state changed within GPU RNG context
# CPU RNG state changed within GPU RNG context
# CPU RNG state changed within GPU RNG context
# CPU RNG state changed within GPU RNG context
# [NeMo I 2024-11-28 09:58:12 megatron_parallel:524]  > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 26313472
# [NeMo W 2024-11-28 09:58:12 nemo_logging:349] /opt/megatron-lm/megatron/core/dist_checkpointing/strategies/torch.py:755: FutureWarning: `load_state_dict` is deprecated and will be removed in future versions. Please use `load` instead.
#       checkpoint.load_state_dict(

# [NeMo W 2024-11-28 09:58:12 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/torch/distributed/checkpoint/planner_helpers.py:311: FutureWarning: Please use DTensor instead and we are deprecating ShardedTensor.
#       device = getattr(value, "device", None)

# GPU available: True (cuda), used: True
# TPU available: False, using: 0 TPU cores
# HPU available: False, using: 0 HPUs
# GPU available: True (cuda), used: True
# TPU available: False, using: 0 TPU cores
# HPU available: False, using: 0 HPUs
# GPU available: True (cuda), used: True
# TPU available: False, using: 0 TPU cores
# HPU available: False, using: 0 HPUs

# While import_ckpt() shows a line that "checkpoint created on /tmp/...",
# export_ckpt() just doesn't show this line. But worry not, we can check.
find '/tmp/hf-ckpt-from-nemo/llama3-haha' -type f | xargs ls -alh
# -rw-r--r-- 1 root root  669 Nov 28 09:58 /tmp/hf-ckpt-from-nemo/llama3-haha/config.json
# -rw-r--r-- 1 root root  111 Nov 28 09:58 /tmp/hf-ckpt-from-nemo/llama3-haha/generation_config.json
# -rw-r--r-- 1 root root 446K Nov 28 09:58 /tmp/hf-ckpt-from-nemo/llama3-haha/merges.txt
# -rw-r--r-- 1 root root  51M Nov 28 09:58 /tmp/hf-ckpt-from-nemo/llama3-haha/model.safetensors
# -rw-r--r-- 1 root root  730 Nov 28 09:58 /tmp/hf-ckpt-from-nemo/llama3-haha/special_tokens_map.json
# -rw-r--r-- 1 root root  580 Nov 28 09:58 /tmp/hf-ckpt-from-nemo/llama3-haha/tokenizer_config.json
# -rw-r--r-- 1 root root 976K Nov 28 09:58 /tmp/hf-ckpt-from-nemo/llama3-haha/vocab.json

# Let's compare the size to the .nemo format.
find '/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last' -type f | xargs ls -alh
# -rw-r--r-- 1 root root   584 Nov 28 09:46 '/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/context/1e5170a3-ff33-49d7-b96c-a98dec202215'
# -rw-r--r-- 1 root root   179 Nov 28 09:46 '/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/context/b5ec73b5-2eb6-41bf-82bf-904aabb7879a'
# -rw-r--r-- 1 root root   62K Nov 28 09:46 '/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/context/io.json'
# -rw-r--r-- 1 root root  446K Nov 28 09:46 '/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/context/megatron-gpt-345m_merges'
# -rw-r--r-- 1 root root 1018K Nov 28 09:46 '/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/context/megatron-gpt-345m_vocab'
# -rw-r--r-- 1 root root  6.1K Nov 28 09:46 '/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/context/model.yaml'
# -rw-r--r-- 1 root root  4.2K Nov 28 09:46 '/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/weights/.metadata'
# -rw-r--r-- 1 root root   26M Nov 28 09:46 '/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/weights/__0_0.distcp'
# -rw-r--r-- 1 root root   26M Nov 28 09:46 '/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/weights/__0_1.distcp'
# -rw-r--r-- 1 root root  2.1K Nov 28 09:46 '/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/weights/common.pt'
# -rw-r--r-- 1 root root   119 Nov 28 09:46 '/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/weights/metadata.json'
