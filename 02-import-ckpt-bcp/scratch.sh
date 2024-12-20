#!/bin/bash

## This is not meant to be an executable. The .sh extension is for syntax highlighting.


################################################################################
# 000: create an interactive session on DGX Cloud using nemo:24.09 container.
#
# NOTE: if you're not using DGX Cloud, skip this section, and directly proceed
#       to the next one (section 010).
################################################################################
# 0       : Use stock ngc cli to submit a job.
# Non-zero: Use our fancy cli (which wrap ngc cli) to submit a job.
#           Will display job url and recommended ngc cli to execute afterwards.
#           Need https://github.com/verdimrc/pyutil-02/blob/main/bin/ngc-base-command-job-run
#           in your $PATH
: "${FANCY_CLI:=1}"
: "${JOB_NAME:=nemo-2.0-import-ckpt-cli}"

# https://github.com/verdimrc/pyutil-02/blob/main/bin/assumingc
. assumingc xxx/yyy

declare -a JOB_ARGS=(
    # NGC CLI 3.54.0 (and possibly earlier) has new syntax for time units

    --result /result
    --image "nvcr.io/nvidia/nemo:24.09"

    # This line mount xxxx into the container as /root/.cache
    --workspace xxxx:/root/.cache:RW

    ############################################################################
    ## Uncomment below for importing llama3_8b.
    ############################################################################
    # --instance dgxa100.80g.1.norm    # 1x A100-80GB, 128GB RAM
    # --min-timeslice 24H
    # --total-runtime 24H
    ############################################################################


    ############################################################################
    ## Uncomment below for llama3-70b.
    ## Also, stay humble with the request, otherwise I suspect long-period job
    ## will get killed almost immediately.
    ############################################################################
    --instance dgxa100.80g.4.norm   # Provides 512GB RAM. Peak usage: 417 GB.
    --min-timeslice 1H
    --total-runtime 1H
    --priority LOW
    ############################################################################
)

if [[ ${FANCY_CLI} == 0 ]]; then
    ngc base-command job run --name "$JOB_NAME" --commandline 'sleep infinity' "${JOB_ARGS[@]}"
elif command -v ngc-base-command-job-run 2>&1 >/dev/null ; then
    ngc-base-command-job-run --name "$JOB_NAME" "${JOB_ARGS[@]}"
else
    echo "Fancy wrapper not found. Fallback to stock cli..."
    ngc base-command job run --name "$JOB_NAME" --commandline 'sleep infinity' "${JOB_ARGS[@]}"
fi

## Unit-test parser.
# test_job_submit_output='
# ----------------------------------------------------------
#  Job Information
#    Id: 7180359
#    Name: nemo-2.0-import-ckpt-cli
#    Number of Replicas: 1
#    Job Type: BATCH
#    Submitted By: Verdi March
#    Order: 50
#    Priority: NORMAL
#  Job Container Information
#    Docker Image URL: nvidia/nemo:24.09
#  Job Commands
#    Command: sleep inifinity
#  Datasets, Workspaces and Results
#      Workspace ID: xxxx
#        Workspace Name: yyyy
#        Workspace Mount Point: /root/.cache
#        Workspace Mount Mode: RW
#      Result Mount Point: /result
#  Job Resources
#    Instance Type: dgxa100.80g.1.norm
#    Instance Details: 1 GPU, 11.0 CPU, 128 GB System Memory
#    ACE: xxx
#    Team: yyy
#  Job Labels
#    Locked: False
#  Job Status
#    Created at: 2024-11-25 07:59:10 UTC
#    Status: CREATED
#    Preempt Class: RUNONCE
#    Total Runtime: 02H00M00S
#    Minimum Timeslice: 02H00M00S
# ----------------------------------------------------------'
# echo "$test_job_submit_output" | grep 'Id: ' -m 1 | awk '{print $2}'

exit $?


################################################################################
# 010: Within container (root)
################################################################################
apt update

# Optional: useful CLIs
apt install -y tmux dstat htop time tree bat
ln -s /usr/bin/batcat /usr/local/bin/bat
export DSTAT_OPTS=-cdngym

# Don't know why bcp suddenly needs this.
export PYTHONPATH=/opt/NeMo:$PYTHONPATH

# Verify we load nemo from /opt/NeMo
python -c 'import nemo; print(f"{nemo.__file__}")'
# PASS: /opt/NeMo/nemo/__init__.py
# FAIL: empty line

tmux   # Or: tmux attach (as appropriate).

### Below onwards are under a tmux session. ###

# mkdir -p /tmp/checkpoints/   # Not needed. Nemo will auto-create the output_path.

# Make sure you've huggingface-cli login
# Tell-tale sign: /root/.cache/huggingface/{token,stored_tokens}
# NOTE: on a rootless container, the HF cache defaults to $HOME/.cache/huggingface.

# NOTE: ensure /root/.cache/ within the container has the .py script.

# Warm start: HF checkpoint already cached on local fs
/usr/bin/time python /root/.cache/convert-ckpt-short.py
# ...
#
# ...
# 407.29user 237.85system 2:13.94elapsed 481%CPU (0avgtext+0avgdata 16498304maxresident)k
# 9832512inputs+33585304outputs (1158083major+1305985minor)pagefaults 0swaps
#
tree /tmp/checkpoints/
# /tmp/checkpoints/
# ├── context
# │   ├── io.json
# │   ├── model.yaml
# │   └── nemo_tokenizer
# │       ├── special_tokens_map.json
# │       ├── tokenizer.json
# │       ├── tokenizer.model
# │       └── tokenizer_config.json
# └── weights
#     ├── __0_0.distcp
#     ├── __0_1.distcp
#     ├── common.pt
#     └── metadata.json
#
## NOTES: for clear naming, set output_path to /tmp/checkpoints/xxxx.
#
du -shc /tmp/checkpoints/
# 16G     /tmp/checkpoints/
# 16G     total
#
find /tmp/checkpoints -type f | xargs ls -alh
# -rw-r--r-- 1 root root  54K Nov 26 08:19 /tmp/checkpoints/context/io.json
# -rw-r--r-- 1 root root 5.7K Nov 26 08:19 /tmp/checkpoints/context/model.yaml
# -rw-r--r-- 1 root root  636 Nov 26 08:18 /tmp/checkpoints/context/nemo_tokenizer/special_tokens_map.json
# -rw-r--r-- 1 root root  33M Nov 26 08:18 /tmp/checkpoints/context/nemo_tokenizer/tokenizer.json
# -rw-r--r-- 1 root root 4.1M Nov 26 08:18 /tmp/checkpoints/context/nemo_tokenizer/tokenizer.model
# -rw-r--r-- 1 root root  40K Nov 26 08:18 /tmp/checkpoints/context/nemo_tokenizer/tokenizer_config.json
# -rw-r--r-- 1 root root  40K Nov 26 08:19 /tmp/checkpoints/weights/.metadata
# -rw-r--r-- 1 root root 7.9G Nov 26 08:19 /tmp/checkpoints/weights/__0_0.distcp
# -rw-r--r-- 1 root root 7.9G Nov 26 08:19 /tmp/checkpoints/weights/__0_1.distcp
# -rw-r--r-- 1 root root 2.8K Nov 26 08:19 /tmp/checkpoints/weights/common.pt
# -rw-r--r-- 1 root root  119 Nov 26 08:19 /tmp/checkpoints/weights/metadata.json

# Below is rerun, but patch the .py file to set output_path=None.
/usr/bin/time python /root/.cache/convert-ckpt-short.py
# ...
# Converted Gemma model to Nemo, model saved to /root/.cache/nemo/models/google/gemma-2b
# ...
# 313.14user 149.06system 1:24.26elapsed 548%CPU (0avgtext+0avgdata 16493096maxresident)k
# 16inputs+33566440outputs (523major+2436760minor)pagefaults 0swaps
#
tree /root/.cache/nemo/
#
/root/.cache/nemo/
# ├── datasets
# │   └── squad
# │       ├── test.jsonl
# │       ├── training.jsonl
# │       └── validation.jsonl
# └── models
#     ├── google
#     │   └── gemma-2b
#     │       ├── context
#     │       │   ├── io.json
#     │       │   ├── model.yaml
#     │       │   └── nemo_tokenizer
#     │       │       ├── special_tokens_map.json
#     │       │       ├── tokenizer.json
#     │       │       ├── tokenizer.model
#     │       │       └── tokenizer_config.json
#     │       └── weights
#     │           ├── .metadata
#     │           ├── __0_0.distcp
#     │           ├── __0_1.distcp
#     │           ├── common.pt
#     │           └── metadata.json
#     └── meta-llama
#         └── Meta-Llama-3-8B

# Let's do llama3 7b now. Patch the .py file accordingly.
# Warm start: HF checkpoint already cached on local fs
/usr/bin/time python /root/.cache/convert-ckpt-short.py
# ...
# Converted Llama model to Nemo, model saved to /tmp/checkpoints/meta-llama/Meta-Llama-3-8B
# ...
# 1754.13user 610.01system 4:24.68elapsed 893%CPU (0avgtext+0avgdata 50399260maxresident)k
# 31386216inputs+92308640outputs (3784954major+1458703minor)pagefaults 0swaps
#
find /tmp/checkpoints/meta-llama/Meta-Llama-3-8B -type f | xargs ls -alh
# -rw-r--r-- 1 root root  54K Nov 26 08:49 /tmp/checkpoints/meta-llama/Meta-Llama-3-8B/context/io.json
# -rw-r--r-- 1 root root 5.7K Nov 26 08:49 /tmp/checkpoints/meta-llama/Meta-Llama-3-8B/context/model.yaml
# -rw-r--r-- 1 root root  301 Nov 26 08:47 /tmp/checkpoints/meta-llama/Meta-Llama-3-8B/context/nemo_tokenizer/special_tokens_map.json
# -rw-r--r-- 1 root root  17M Nov 26 08:47 /tmp/checkpoints/meta-llama/Meta-Llama-3-8B/context/nemo_tokenizer/tokenizer.json
# -rw-r--r-- 1 root root 4.1M Nov 26 08:27 /tmp/checkpoints/meta-llama/Meta-Llama-3-8B/context/nemo_tokenizer/tokenizer.model
# -rw-r--r-- 1 root root  50K Nov 26 08:47 /tmp/checkpoints/meta-llama/Meta-Llama-3-8B/context/nemo_tokenizer/tokenizer_config.json
# -rw-r--r-- 1 root root  68K Nov 26 08:49 /tmp/checkpoints/meta-llama/Meta-Llama-3-8B/weights/.metadata
# -rw-r--r-- 1 root root  22G Nov 26 08:49 /tmp/checkpoints/meta-llama/Meta-Llama-3-8B/weights/__0_0.distcp
# -rw-r--r-- 1 root root  22G Nov 26 08:49 /tmp/checkpoints/meta-llama/Meta-Llama-3-8B/weights/__0_1.distcp
# -rw-r--r-- 1 root root 2.8K Nov 26 08:49 /tmp/checkpoints/meta-llama/Meta-Llama-3-8B/weights/common.pt
# -rw-r--r-- 1 root root  119 Nov 26 08:49 /tmp/checkpoints/meta-llama/Meta-Llama-3-8B/weights/metadata.json


# Let's do llama3 70b now. Patch the .py file accordingly.
# - change model source to llama3-70b
# - change output path to /result. Do not use /tmp, because this is backed by tmpfs (hence, taking
#   away RAM.)
#
# Cold start: include HF checkpoint download. Took 56 mins download all the 30x
#             .safetensors files, each at 4.66 GB.
#
# Peak memory usage: 473 GB, so make sure to use a larger instance with sufficient RAM.
#
# JOB_NAME='import-llama3-70b' ./scratch.sh


# Below are cold start, getting killed by OOM, but nevertheless it tells us the downloading time!
/usr/bin/time python /root/.cache/convert-ckpt-short.py
# ...
# model-00030-of-00030.safetensors: 100%|███████████████████████████████████████████████████████████| 2.10G/2.10G [00:49<00:00, 42.4MB/s]
# Downloading shards: 100%|██████████████████████████████████████████████████████████████████████| 30/30 [55:56<00:00, 111.90s/it].3MB/s]
# Loading checkpoint shards:  47%|█████████████████████████████████▏                                     | 14/30 [05:15<06:24, 24.06s/it]
# Command terminated by signal 9
# 682.73user 3066.62system 1:01:30elapsed 101%CPU (0avgtext+0avgdata 133826500maxresident)k
# 132950272inputs+275600816outputs (16502191major+10029816minor)pagefaults 0swaps
#
## Yup, OOM. So, we need to rerun the job on a larger instance.


# Below are importing llama3-70b, warm start.
#
# Watch RAM activity with dstat.
# Watch fs usage: watch -n4 ls -alh /result/meta-llama/Meta-Llama-3-70B/weights/
/usr/bin/time python /root/.cache/convert-ckpt-short.py
# ...
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████| 30/30 [07:38<00:00, 15.30s/it]
# generation_config.json: 100%|████████████████████████████████████████████████████████████████| 177/177 [00:00<00:00, 1.70MB/s]
# tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████| 50.6k/50.6k [00:00<00:00, 4.28MB/s]
# tokenizer.json: 100%|████████████████████████████████████████████████████████████████████| 9.09M/9.09M [00:00<00:00, 24.2MB/s]
# special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████| 73.0/73.0 [00:00<00:00, 1.18MB/s]
# ...
# Converted Llama model to Nemo, model saved to /result/meta-llama/Meta-Llama-3-70B
# ...
# 5609.20user 7154.81system 17:28.10elapsed 1217%CPU (0avgtext+0avgdata 449819964maxresident)k
# 275659328inputs+845032920outputs (34186769major+23325501minor)pagefaults 0swap

ls -alh /result/meta-llama/Meta-Llama-3-70B/weights/
# total 403G
# drwxr-xr-x 2 99 99 4.0K Nov 27 08:26 .
# drwxr-xr-x 4 99 99 4.0K Nov 27 08:26 ..
# -rw-r--r-- 1 99 99 165K Nov 27 08:26 .metadata
# -rw-r--r-- 1 99 99 202G Nov 27 08:26 __0_0.distcp
# -rw-r--r-- 1 99 99 202G Nov 27 08:26 __0_1.distcp
# -rw-r--r-- 1 99 99 2.8K Nov 27 08:19 common.pt
# -rw-r--r-- 1 99 99  119 Nov 27 08:26 metadata.json
#
ls -al /result/meta-llama/Meta-Llama-3-70B/weights/
# total 422403188
# drwxr-xr-x 2 99 99         4096 Nov 27 08:26 .
# drwxr-xr-x 4 99 99         4096 Nov 27 08:26 ..
# -rw-r--r-- 1 99 99       168717 Nov 27 08:26 .metadata
# -rw-r--r-- 1 99 99 216270054616 Nov 27 08:26 __0_0.distcp
# -rw-r--r-- 1 99 99 216270020668 Nov 27 08:26 __0_1.distcp
# -rw-r--r-- 1 99 99         2839 Nov 27 08:19 common.pt
# -rw-r--r-- 1 99 99          119 Nov 27 08:26 metadata.json
#
# Compare to original HF size.
find /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-70B/ -type f | du -h
# 403G    ./meta-llama/Meta-Llama-3-70B/weights
# 17M     ./meta-llama/Meta-Llama-3-70B/context/nemo_tokenizer
# 17M     ./meta-llama/Meta-Llama-3-70B/context
# 403G    ./meta-llama/Meta-Llama-3-70B
# 403G    ./meta-llama
# 403G    .
#
rm /result/meta-llama/     # Before terminating our job, free 400GB disk space :)


## Clear tokens from the workspace (since it's mounted as /root/.cache), so we want to prevent
## the token to stay long-term on persistent storage.
huggingface-cli logout
