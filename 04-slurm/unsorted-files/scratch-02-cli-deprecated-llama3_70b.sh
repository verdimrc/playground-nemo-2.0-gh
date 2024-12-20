#!/bin/bash

################################################################################
# Deprecated.
#
# For some reasons, overriding recipe via cli results in weird error (tokenizer
# file not found, yada 2x.). Not sure if this is the true cause or not.
################################################################################


# Treat this file as a doc, not as an executable.
# The extension .sh is for syntax highlighting.


################################################################################
# 000: start the client continer to submit a Slurm job.
################################################################################
docker pull nvcr.io/nvidia/nemo:24.09
enroot import -o /mnt/myshareddir/nvcr.io__nvidia__nemo__24.09.sqsh dockerd://nvcr.io/nvidia/nemo:24.09

# "enroot create then start" to avoid hang when directly "enroot start xxx.sqsh".
# See: https://gist.github.com/verdimrc/d6f0c5f0f66debb5383f66ed9ac3da31#1-quickrun
enroot create /mnt/myshareddir/nvcr.io__nvidia__nemo__24.09.sqsh
# Will "extract" to /mnt/myshareddir/nvidia-verdi/.local/share/enroot/nvcr.io__nvidia__nemo__24.09
# where ~/.local/share/enroot is the ENROOT_XXX env var (see /etc/xxx/enroot.conf).

enroot list

mkdir -p /mnt/myshareddir/nvidia-verdi/nemo-workdir/

# To test additional mount points defined in the .py files.
mkdir -p /mnt/myshareddir/nvidia-verdi/testdir-01
mkdir -p /mnt/myshareddir/nvidia-verdi/testdir-02
touch /mnt/myshareddir/nvidia-verdi/testdir-01/01.txt
touch /mnt/myshareddir/nvidia-verdi/testdir-02/02.txt

# These are mount point for the client container.
declare -a MOUNT_ARGS=(
    # The source code of the job.
    -m /mnt/myshareddir/nvidia-verdi/playground-nemo-2.0:/haha

    # For passwordless ssh to the Slurm login node.
    -m /mnt/myshareddir/nvidia-verdi/.ssh:/root/.ssh

    # OPTIONAL: my client machine = Slurm login node, so I like to have the ability
    # to inspect the generated artifacts directly from the client container.
    #
    # Not needed when client machine != Slurm login node, because the lhs is the
    # shared fs path on the Slurm cluster. In fact, the client machine won't have
    # the lhs path, so enroot will error when trying to start mount this
    # non-existent path.
    -m /mnt/myshareddir/nvidia-verdi/nemo-workdir:/nmwd
)

# NOTE: no .sqsh.
enroot start --root --rw "${MOUNT_ARGS[@]}" nvcr.io__nvidia__nemo__24.09


################################################################################
# 010: Inside client container: generate Slurm script, then submit as a Slurm job.
################################################################################
declare -a ARGS=(
    ## num_nodes must match my_slurm_executor().
    ##
    ## dir is the path under the job container, and we choose /hehe which is
    ## a container mountpoint defined in the .py script.
    --factory "llama3_70b(name='HEHE',num_nodes=2,dir='/hehe/checkpoints/llama3')"
    # --executor my_slurm_executor  # This is in main, but not yet in 24.09

    trainer.max_steps=30
    trainer.val_check_interval=30  # HAHA: already the default.

    # llama3_70b recipes has these default settings:
    #     tp=4, pp=4, cp=2, num_nodes=4, num_gpus_per_node=8.
    #     See: /opt/NeMo/nemo/collections/llm/recipes/llama3_70b.py (files under container).
    #
    # We change to two nodes, because the remaining two seems to be down (incorrect pyxis).
    # trainer.num_nodes=2
    trainer.strategy.context_parallel_size=1
    trainer.limit_val_batches=10
)

# BIN_DIR=$(dirname `realpath ${BASH_SOURCE[0]}`) || BIN_DIR="."
BIN_DIR=/haha
# python $BIN_DIR/06-slurm/default_executor_02.py "${ARGS[@]}" --dryrun "$@"  ## => error: no scripts generated!

python $BIN_DIR/06-slurm/default_executor_02.py "${ARGS[@]}" "$@"
        # [17:08:28] INFO     Launched app: slurm_tunnel://nemo_run/190
        # ...
        #     invoke.exceptions.UnexpectedExit: Encountered a bad command exit code!

        #    Command: 'sacct --parsable2 -j 132'

        #    Exit code: 1

        #    Stdout:



        #    Stderr:

        #    Slurm accounting storage is disabled

# Yeah, it's a bit tricky to figure out where the generated file. The best I can find is to
# go to SLurm login node, then scontrol show job 190.

# Then edit the files, either now or during the next step (on Slurm login node).
# - to disable the accounting, because my Slurm testbed has not setup accounting.
# - add export HF_HOME=/tmp/haha.cache/ ## Only when the .py file doesn't have this already.
# - add export TORCH_HOME=/tmp/haha.cache/  ## Only when the .py file doesn't have this already.
# - add a few other env vars to control the cache dirs.  ## Only when the .py file doesn't have this already.

# Below is path on Slurm login node. Note that on my test setting, /mnt/myshareddir/nvidia-verdi/nemo-workdir is /nmwd.
scontrol show job 190
# /mnt/myshareddir/nvidia-verdi/nemo-workdir/nemo.collections.llm.api.pretrain/nemo.collections.llm.api.pretrain_1734432039/nemo.collections.llm.api.pretrain/sbatch_asdf-asdf.nemo.collections.llm.api.pretrain_<JOB_ID>.out
# /mnt/myshareddir/nvidia-verdi/nemo-workdir/nemo.collections.llm.api.pretrain/nemo.collections.llm.api.pretrain_1734432039/nemo.collections.llm.api.pretrain/
# /mnt/myshareddir/nvidia-verdi/nemo-workdir/nemo.collections.llm.api.pretrain/nemo.collections.llm.api.pretrain_1734432039/nemo.collections.llm.api.pretrain_sbatch.sh

du -shc /nmwd/exp/HEHE/checkpoints/*


################################################################################
# 020: On Slurm login node: resubmit the sbatch script.
################################################################################
# On host, submit job. REMINDER: make sure you've modified the sbatch file to apply WARs.
#
# The changes MUST be done on sbatch script from the login node, and NOT from the
# client machine (unless the client machine is the same as the login node).

sbatch --requeue --parsable \
/mnt/myshareddir/nvidia-verdi/nemo-workdir/nemo.collections.llm.api.pretrain/nemo.collections.llm.api.pretrain_1734432039/nemo.collections.llm.api.pretrain_sbatch.sh

####
exit "$?"
scontrol show job 191

cd /mnt/myshareddir/nvidia-verdi/nemo-workdir/nemo.collections.llm.api.pretrain/nemo.collections.llm.api.pretrain_1734426502/
cd nemo.collections.llm.api.pretrain/

##
srun --container-image /mnt/myshareddir/nvcr.io__nvidia__nemo__24.09.sqsh --container-mounts /mnt/myshareddir/nvidia-verdi/nemo-workdir/nemo.collections.llm.api.pretrain/nemo.collections.llm.api.pretrain_1734347151/nemo.collections.llm.api.pretrain:/nemo_run --container-workdir /nemo_run/code --wait=60 --kill-on-bad-exit=1 --pty /bin/bash

export PYTHONPATH=/nemo_run/NeMo:$PYTHONPATH
python -m nemo_run.core.runners.fdl_runner -n nemo.collections.llm.api.pretrain -p /nemo_run/configs/nemo.collections.llm.api.pretrain_packager /nemo_run/configs/nemo.collections.llm.api.pretrain_fn_or_script


AUTOTOKENIZER.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name,
                vocab_file=vocab_file,
                merges_file=merges_file,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
            )

from transformers import AutoTokenizer as AUTOTOKENIZER
pretrained_model_name='gpt2'
use_fast=False
trust_remote_code=False
vocab_file='/tmp/haha.cache/torch/megatron/megatron-gpt-345m_vocab'
merges_file='/tmp/haha.cache/torch/megatron/megatron-gpt-345m_merges'

tokenizer = AUTOTOKENIZER.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name,
    # vocab_file=vocab_file,
    # merges_file=merges_file,
    use_fast=use_fast,
    trust_remote_code=trust_remote_code,
)


/mnt/myshareddir/nvidia-verdi/nemo-workdir/nemo.collections.llm.api.pretrain/nemo.collections.llm.api.pretrain_1734432922/nemo.collections.llm.api.pretrain/log-asdf-asdf.nemo.collections.llm.api.pretrain_238_0.out
#export HF_HOME=/tmp/haha.cache/huggingface
export HF_HOME=/hihi/.cache/huggingface    # Multi-node needs this on shared dir
+ export HF_HOME=/hihi/.cache/huggingface
+ HF_HOME=/hihi/.cache/huggingface
export TORCH_HOME=/hihi/.cache/torch       # Multi-node needs this on shared dir (for MEGATRON_CACHE)
+ export TORCH_HOME=/hihi/.cache/torch
+ TORCH_HOME=/hihi/.cache/torch
export NEMO_HOME=/hihi/.cache/nemo         # Multi node needs this on shared dir
+ export NEMO_HOME=/hihi/.cache/nemo
+ NEMO_HOME=/hihi/.cache/nemo
export TRITON_CACHE_DIR=/tmp/haha.cache/.triton
+ export TRITON_CACHE_DIR=/tmp/haha.cache/.triton
+ TRITON_CACHE_DIR=/tmp/haha.cache/.triton


###/mnt/myshareddir/nvidia-verdi/nemo-workdir/nemo.collections.llm.api.pretrain/nemo.collections.llm.api.pretrain_1734432922/nemo.collections.llm.api.pretrain/log-asdf-asdf.nemo.collections.llm.api.pretrain_239_0.out
/mnt/myshareddir/nvidia-verdi/nemo-workdir/nemo.collections.llm.api.pretrain/nemo.collections.llm.api.pretrain_1734441921/nemo.collections.llm.api.pretrain/log-asdf-asdf.nemo.collections.llm.api.pretrain_241_0.out
export HF_HOME=/tmp/haha.cache/huggingface
+ export HF_HOME=/tmp/haha.cache/huggingface
+ HF_HOME=/tmp/haha.cache/huggingface
#export HF_HOME=/hihi/.cache/huggingface    # Multi-node needs this on shared dir
export TORCH_HOME=/hihi/.cache/torch       # Multi-node needs this on shared dir (for MEGATRON_CACHE)
+ export TORCH_HOME=/hihi/.cache/torch
+ TORCH_HOME=/hihi/.cache/torch
export NEMO_HOME=/hihi/.cache/nemo         # Multi node needs this on shared dir???
+ export NEMO_HOME=/hihi/.cache/nemo
+ NEMO_HOME=/hihi/.cache/nemo
export TRITON_CACHE_DIR=/tmp/haha.cache/.triton
+ export TRITON_CACHE_DIR=/tmp/haha.cache/.triton
+ TRITON_CACHE_DIR=/tmp/haha.cache/.triton
