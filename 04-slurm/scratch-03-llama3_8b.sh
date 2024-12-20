#!/bin/bash

# Treat this file as a doc, not as an executable.
# The extension .sh is for syntax highlighting.


################################################################################
# 000: start the client container to submit a Slurm job.
################################################################################
docker pull nvcr.io/nvidia/nemo:24.09
enroot import -o /mnt/myshareddir/nvcr.io__nvidia__nemo__24.09.sqsh dockerd://nvcr.io/nvidia/nemo:24.09

# "enroot create then start" to avoid hang when directly "enroot start xxx.sqsh".
# See: https://gist.github.com/verdimrc/d6f0c5f0f66debb5383f66ed9ac3da31#1-quickrun
enroot create /mnt/myshareddir/nvcr.io__nvidia__nemo__24.09.sqsh

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
    -m /mnt/myshareddir/nvidia-verdi/nemo-workdir:/nmwd

    # OPTIONAL: my client machine = Slurm login node, so I like to have the ability
    # to inspect the generated artifacts directly from the client container.
    #
    # Not needed when client machine != Slurm login node, because the lhs is the
    # shared fs path on the Slurm cluster. In fact, the client machine won't have
    # the lhs path, so enroot will error when trying to start mount this
    # non-existent path.
    -m /mnt/myshareddir/nvidia-verdi/.ssh:/root/.ssh
)

# NOTE: no .sqsh.
enroot start --root --rw "${MOUNT_ARGS[@]}" nvcr.io__nvidia__nemo__24.09


################################################################################
# 010: Inside client container: generate Slurm script, then submit as a Slurm job.
################################################################################

declare -a ARGS=(
    ## num_nodes must match my_slurm_executor() in the .py file.
    --factory "llama3_8b(name='HEHE',num_nodes=2,dir='/hehe/checkpoints/llama3')"
    # --executor my_slurm_executor  # This is in main, but not yet in 24.09

    trainer.max_steps=30
    trainer.val_check_interval=30  # HAHA: already the default.
    trainer.devices=8  # gpus_per_node
    # trainer.strategy.tensor_model_parallel_size=1
    trainer.strategy.tensor_model_parallel_size=2
    trainer.strategy.context_parallel_size=1

    trainer.limit_val_batches=10
)

# BIN_DIR=$(dirname `realpath ${BASH_SOURCE[0]}`) || BIN_DIR="."
BIN_DIR=/haha
# python $BIN_DIR/06-slurm/default_executor_03.py "${ARGS[@]}" --dryrun "$@"  ## => error: no scripts generated!

python $BIN_DIR/06-slurm/default_executor_03.py "${ARGS[@]}" "$@"
        #     invoke.exceptions.UnexpectedExit: Encountered a bad command exit code!

        #    Command: 'sacct --parsable2 -j 132'

        #    Exit code: 1

        #    Stdout:



        #    Stderr:

        #    Slurm accounting storage is disabled

# Yeah, it's a bit tricky to figure out where the generated file. The best I can find is to
# go to SLurm login node, then scontrol show job 190, and figure the directory based on the
# stdout or stderr path.
#
# If `scontrol show job 190` shows nothing, chances are you do not setup accounting. In this case,
# goto the job_remote_workdir (which in my case is /mnt/myshareddir/nvidia-verdi/nemo-workdir),
# then do a `find . -name 'log_*_<JOB_ID>_0.out` and you should be able to figure out the
# correct directory, e.g,
# /mnt/myshareddir/nvidia-verdi/nemo-workdir/**/nemo.collections.llm.api.pretrain_1734454315/
#
# Then edit the files, either now or during the next step (on Slurm login node).
# - to disable the accounting, because my Slurm testbed has not setup accounting.


################################################################################
# 020: On Slurm login node: resubmit the sbatch script.
################################################################################
# On host, submit job. REMINDER: make sure you've modified the sbatch file to
# disable accounting.
#
# The changes MUST be done on sbatch script from the login node, and NOT from the
# client machine (unless the client machine is the same as the login node).

sbatch --requeue --parsable \
/mnt/myshareddir/nvidia-verdi/nemo-workdir/nemo.collections.llm.api.pretrain/nemo.collections.llm.api.pretrain_1734442528/nemo.collections.llm.api.pretrain_sbatch.sh

exit "$?"


################################################################################
# 030: Sample commands on login node to navigate job dirs, files, etc.
################################################################################

scontrol show job 136

cd /mnt/myshareddir/nvidia-verdi/nemo-workdir/nemo.collections.llm.api.pretrain/nemo.collections.llm.api.pretrain_1734347151/
cd nemo.collections.llm.api.pretrain/

# In case I want to open an interactive session on a compute node, using the
# artifacts of a specific node.
#
# I also shadow the /opt/NeMo in the enroot container with one from my host fs -> notice the
# container mounts.
srun --container-image /mnt/myshareddir/nvcr.io__nvidia__nemo__24.09.sqsh --container-mounts /mnt/myshareddir/nvidia-verdi/nemo-workdir/nemo.collections.llm.api.pretrain/nemo.collections.llm.api.pretrain_1734347151/nemo.collections.llm.api.pretrain:/nemo_run --container-workdir /nemo_run/code --wait=60 --kill-on-bad-exit=1 --pty /bin/bash
#
## Below is inside the container: because I mount the host's NeMo dir NOT to /opt/NeMo, I ask python to
## use the host's version by setting the PYTHON PATH
export PYTHONPATH=/nemo_run/NeMo:$PYTHONPATH
python -m nemo_run.core.runners.fdl_runner -n nemo.collections.llm.api.pretrain -p /nemo_run/configs/nemo.collections.llm.api.pretrain_packager /nemo_run/configs/nemo.collections.llm.api.pretrain_fn_or_script
