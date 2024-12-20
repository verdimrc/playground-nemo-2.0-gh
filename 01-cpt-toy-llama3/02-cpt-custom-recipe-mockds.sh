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

rm -fr /tmp/exp/
mkdir -p /tmp/exp/

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_FUSED_ATTN=1
export NCCL_DEBUG=INFO

## Uncomment below, to validate nemo picks-up this env var.
## Expected outcome: set to invalid GPU index, e.g., laptop with 1 gpu => index>0 will crash.
# export CUDA_VISIBLE_DEVICES=1

rm -fr /tmp/ftexp/

declare -a ARGS=(
    ## Factory defined in the .py script.
    #--factory "llama3_8b(name='HEHE',num_nodes=$NUM_NODES,dir='/tmp/exp')"

    ## Not including weights/ => This will cause optimizer missing error, because default restore_config='nemo://meta-llama/Meta'
    ## THIS is wrong for continual xxx. The stanza is left here for FYI / historical context only.
    #resume.resume_from_path='/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/'
    ##
    ## ==> THIS IS THE CORRECT WAY TO Continual xxx.
    resume.restore_config.path="/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/"

    resume.resume_if_exists=True

    # Make sure to use "xxx=xxx", and not "xxx = xxx". After all, this is a bash script, not python.
    model.config.seq_length=128
    model.config.num_layers=1
    model.config.hidden_size=256
    model.config.ffn_hidden_size=512
    model.config.num_attention_heads=32

    data.seq_length=128   # Make sure to match model.config.seq_length
    data.global_batch_size=64
    # data.num_workers=0

    trainer.max_steps=30
    trainer.val_check_interval=30  # HAHA: already the default.
    trainer.devices=1
    trainer.strategy.tensor_model_parallel_size=1
    trainer.strategy.context_parallel_size=1

    trainer.limit_val_batches=10

    ## Enable as you see it fits to your use case.
    #log.ckpt.save_on_train_epoch_end=True
    #log.ckpt.save_optim_on_train_end=False
)

## Below uses MockDataset.
BIN_DIR=$(dirname `realpath ${BASH_SOURCE[0]}`)
python $BIN_DIR/custom_cpt_recipe.py "${ARGS[@]}" "$@"

du -shc /tmp/exp/HEHE/checkpoints/*
