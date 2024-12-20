# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import nemo_run as run

from nemo.collections import llm

NUM_NODES=4

# This works with docker run (root user:group, which is the default of the container).
def local_executor_torchrun(devices: int = 2) -> run.LocalExecutor:
    env_vars = {
        # "TRANSFORMERS_OFFLINE": "1",   # HAHA: let's just rely on hfhub.
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor


def slurm_executor(
    user: str,
    host: str,
    remote_job_dir: str,
    account: str,
    partition: str,
    nodes: int,
    devices: int,    # gpus_per_node
    time: str = "01:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = "/mnt/myshareddir/nvcr.io__nvidia__nemo__24.09.sqsh",
    retries: int = 0,
) -> run.SlurmExecutor:
    if not (user and host and remote_job_dir and account and partition and nodes and devices):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this function."
        )

    custom_mounts = [] if custom_mounts is None else custom_mounts.copy()

    env_vars = {
        # "TRANSFORMERS_OFFLINE": "1",   # HAHA: we want to still jit pull tokenizers from HFHub
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",

        # Relocate cache dirs to prevent permission errors -- complains about read-only fs.
        # NOTE: all paths are under containers.
        "HF_HOME": "/tmp/haha.cache/huggingface",
        "TORCH_HOME": "/hihi/.cache/torch",   # Multi-node requires this in shared dir
        "NEMO_HOME": "/hihi/.cache/nemo",  # Multi-node requires this in shared dir??
        "TRITON_CACHE_DIR": "/tmp/haha.cache/.triton",   # This is torch dynamo triton, not NVidia triton.
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    # See class SlurmExecutor at
    # https://github.com/NVIDIA/NeMo-Run/blob/main/src/nemo_run/core/execution/slurm.py
    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir,
        ),
        container_image = container_image,
        container_mounts = custom_mounts,
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,
        mem="0",
        exclusive=True,
        gres="gpu:8",

        srun_args=['-l'],

        # Beware, git packager will include only committed files (i.e., versioned files).
        #packager=run.GitArchivePackager(subpath="examples/llm/run"),

        # relative_path is under container. See nemo-run patternpackager doc on the logic,
        # basically this packager will look for files under ${relative_path}/${include_pattern}.
        packager=run.PatternPackager(include_pattern="**", relative_path="/haha/06-slurm"),
    )

    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor


def my_slurm_executor():
    return slurm_executor(
        user="nvidia-verdi",
        host="hgx-056",  # HAHA: localhost will cause local run, without slurm.
        remote_job_dir="/mnt/myshareddir/nvidia-verdi/nemo-workdir",  # Shared fs on host.

        # HAHA: my Slurm testbed doesn't setup sacct, so nemo run will fail. After that, manually
        # change the generated sbatch to disable account, and resubmit.
        account="asdf",

        partition="debug",

        # IMPORTANT: must match your recipe.
        #
        # To avoid DRY between the recipe (with override possibility) and the Slurm executor,
        # it's best to (1) define a custom recipe in this .py file, (2) parse the nodes and
        # gpus_per_node in whatever ways (env vars or cli arg), then (3) instantiate recipe & this
        # slurm executor consistently with the same nodes & gpus_per_node.
        nodes=NUM_NODES,
        devices=8,

        # HAHA: demonstrate mount points. Make sure the lhs paths exist on the shared fs.
        custom_mounts=[
            "/mnt/myshareddir/nvidia-verdi/testdir-01:/hehe",   # checkpoint dir
            "/mnt/myshareddir/nvidia-verdi/testdir-02:/hihi"    # cache dir
        ],
    )


def custom_llama3_70b():
    recipe = llm.recipes.llama3_70b.pretrain_recipe(name='HEHE', num_nodes=NUM_NODES, dir='/hehe/checkpoints/llama3_70b', )
    # recipe.resume.restore_config.xxx
    #restore_config=<Config[RestoreConfig(path='nemo://meta-llama/Meta-Llama-3-8B')]>)]>,

    recipe.trainer.max_steps=30
    # recipe.trainer.val_check_interval=30  # HAHA: already the default? Or does 70b recipe have different default than 8b recipe?
    recipe.trainer.limit_val_batches=10

    # Force ckpt after every step. Use this when debugging checkpointing error, as it triggers
    # checkpointing at every step.
    # See: class ModelCheckpoint.
    #### HAHA: NOTE: JIRA ticket opened. ####
    recipe.log.ckpt.train_time_interval = None  # Undo the recipe's default of every 15min training.
    recipe.log.ckpt.every_n_train_steps = 1     # We want to quickly trigger the error.

    # llama3_70b recipes has these default settings:
    #     tp=4, pp=4, cp=2, num_nodes=4, num_gpus_per_node=8.
    #     See: /opt/NeMo/nemo/collections/llm/recipes/llama3_70b.py (files under container).
    #
    # We change to two nodes, because the remaining two seems to be down (incorrect pyxis).
    # trainer.num_nodes=2
    if NUM_NODES < 4:   # Poor's man feature flag.
        recipe.trainer.strategy.context_parallel_size=1
        recipe.trainer.strategy.activations_checkpoint_granularity='full'
        # IIRC there's another activations hyperparam (like block count or sth) need to set to 1?
        # It was in nemo-1.0 and megatron-lm, at least that's my muddy recollection.

    print(recipe)
    return recipe

if __name__ == "__main__":
    executor=my_slurm_executor()
    run.cli.main(llm.pretrain, default_factory=custom_llama3_70b, default_executor=my_slurm_executor())

    import sys
    sys.exit(0)
    ## Below is from main, and doesn't work on 24.09 due to different API signatures.
    run.cli.main(llm.pretrain, default_executor=local_executor_torchrun)

    # This will re-expose the pretrain entrypoint with your custom local executor as default.

    # To run, for instance, the llama3_8b recipe, use the following command:
    #   python default_executor.py --factory llama3_8b

    # To run with any overrides, use the following command:
    #   python default_executor.py --factory llama3_8b trainer.max_steps=2000

    # To use your custom Slurm executor, use the following command:
    #   python default_executor.py --executor my_slurm_executor --factory llama3_8b
