#!/bin/bash

####
# Treat this file as a doc, not as an executable. The extension .sh is for syntax highlighting.
#
# https://github.com/NVIDIA/NeMo/tree/main/examples/llm/pretrain

# Usage:
#
#     ./scratch.sh
#     ./scratch.sh --no-confirm
#     ./scratch.sh ...

: "${NUM_NODES:=1}"

rm -fr /tmp/checkpoints/llama3/
mkdir -p /tmp/checkpoints/llama3/

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_FUSED_ATTN=1
export NCCL_DEBUG=INFO

## Uncomment below, to validate nemo picks-up this env var.
## Expected outcome: set to invalid GPU index, e.g., laptop with 1 gpu => index>0 will crash.
# export CUDA_VISIBLE_DEVICES=1

declare -a ARGS=(
    --factory "llama3_8b(name='HAHA',num_nodes=$NUM_NODES,dir='/tmp/checkpoints/llama3')"

    # Make sure to use "xxx=xxx", and not "xxx = xxx". After all, this is a bash script, not python.
    model.config.seq_length=128
    model.config.num_layers=1
    model.config.hidden_size=256
    model.config.ffn_hidden_size=512
    model.config.num_attention_heads=32
    data.seq_length=128   # Make sure to match model.config.seq_length
    data.global_batch_size=64
    trainer.max_steps=20
    trainer.val_check_interval=30  # HAHA: already the default.
    trainer.devices=1
    trainer.strategy.tensor_model_parallel_size=1
    trainer.strategy.context_parallel_size=1

    # https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/callbacks/model_checkpoint.py
    #
    # When pretraining < train_time_interval, no optimizer written at the end.
    # This prevents us to test resume, as pretraining with an existing checkpoint
    # (weights only) will fail.
    #
    # Hence, we need to force write the optimizer at the end of the training, so
    # the next pre-training (with a higher trainer.max_steps) can resume.
    log.ckpt.save_optim_on_train_end=True

    # Below ckpt also needs optimizer.
    # resume.resume_from_path='/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/weights/'
)
nemo llm pretrain "${ARGS[@]}" "$@"

du -shc /tmp/checkpoints/llama3/HAHA/checkpoints/*
