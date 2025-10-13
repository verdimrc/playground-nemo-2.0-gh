# Minimalistic BYOC (build your own container) for TE and MBridge

<details><summary>Table of Contents</summary>

- [1. How I run my enroot container](#1-how-i-run-my-enroot-container)
- [2. Inside container](#2-inside-container)
  - [2.1. Build TE wheel](#21-build-te-wheel)
  - [2.2. Install MB](#22-install-mb)
  - [2.3. Verify TE](#23-verify-te)
  - [2.4. Verify MB](#24-verify-mb)
- [3. Appendix: training log](#3-appendix-training-log)
- [4. Appendix: `nemo:25.09.00`](#4-appendix-nemo250900)

</details>

Everything in this document is "*as of this writing*" (Wed 07-Oct-2025). This document is provided
*as-is* as an example, hence adaptations (some portings) are expected.

Intended audience: those who live on tot version of [Megatron-Bridge
(MB)](https://github.com/NVIDIA-NeMo/Megatron-Bridge) and/or [TransformerEngine
(TE)](https://github.com/NVIDIA/TransformerEngine).

TL;DR:

- Base container: `nvcr.io/nvidia/pytorch:25.06-py3`
  ([changelog](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-06.html))
  or `nvcr.io/nvidia/nemo:25.09.xx` (recommended).
- Host clones <https://github.com/NVIDIA-NeMo/Megatron-Bridge.git> and
  <https://github.com/NVIDIA/TransformerEngine.git>.
- Run container, with mounted MB and TE local repos.
- Build TE `.whl`
- Install MB dependencies
- Scrub some of the installed dependencies
- Install TE `.whl`

The remaining document assumes the base container is `nvcr.io/nvidia/pytorch:25.06-py3`.

<details>
<summary>For the chronicle/journal/reasonings of my byoc journey, see <code>haha-*.sh</code>
</summary>

TL;DR:

- Cannot use NGC PyTorch 25.08 or 25.09 as these provides CUDA-13.x
- The stable versions of PyTorch and Triton needs CUDA-12.x.
- Triton TOT adds CUDA-13.x, so at least it's buildable. However, PyTorch-2.9-dev (the version in
  NGC PyTorch 25.09 container) doesn't work with this Triton TOT (version `>3.4`, probably will be
  called `3.5.x`).
- On PyTorch 25.06 container, I found that PT-2.8 doesn't work with Triton 3.4.0, so we downgraded
  one version back to triton-3.3.0.

  TBH, this is just trial-and-error. Run MB, encounter a crash that mentions PyTorch, Dynamo
  compiler, triton, and low-level error (thread _rlock, or invalid triton/cuda function signature,
  etc.).

  Then, we just interatively downgrade the Triton version until finding one that works. Fortunately,
  triton-3.3.0 works, so I didn't have to go back further.

</details>

## 1. How I run my enroot container

Start the base container:

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git

rm /lustre/fsw/general_sa/vmarch/hehe.sqsh
enroot import -o /lustre/fsw/general_sa/vmarch/hehe.sqsh docker://nvcr.io/nvidia/pytorch:25.06-py3
enroot create /lustre/fsw/general_sa/vmarch/hehe.sqsh
enroot start \
    -m /lustre/fsw/general_sa/vmarch/.cache:/tmp/haha.cache \
    -e HF_HOME=/tmp/haha.cache/huggingface \
    -e TORCH_HOME=/tmp/haha.cache/torch \
    -e NEMO_HOME=/tmp/haha.cache/nemo \
    -e TRITON_CACHE_DIR=/tmp/haha.cache/.triton \
    -m $(pwd):/haha \
    -m /lustre/fsw/general_sa/vmarch:/hehe \
    -m $HOME/src/Megatron-Bridge:/opt/Megatron-Bridge \
    -m $HOME/src/TransformerEngine:/opt/TransformerEngine \
    -r -w hehe /bin/bash
```

Follow the [Section 2](#2-inside-container) to build and install your own version of MB and TE.

Once you exit from the container, export to a new `.sqsh` file so you'll have a new container image
with both MB and TE installed.

```bash
rm /lustre/fsw/general_sa/vmarch/hihi.sqsh
enroot export -o /lustre/fsw/general_sa/vmarch/hihi.sqsh hehe

enroot remove hehe
```

At this point, the host filesystem will have these additional files:

- `TransformerEngine/transformer_engine-2.9.0.dev0+7e45be73-cp312-cp312-linux_x86_64.whl`
- `hihi.sqsh` which is the container image with custom MB and TE versions.

## 2. Inside container

NOTE: though this journal doesn't pass --no-cache-dir to pip install, you may want to use this flag
in your actual build.

Pre-requisite:

```bash
export MAX_JOBS=32
unset PIP_CONSTRAINT

# Permanently disable the env var (for the next enroot start of the same enroot create)
sed -i -e "s|^\(PIP_CONSTRAINT=/etc/pip/constraint.txt\)$|#\1|" /etc/environment
```

```console
# Probe dep versions in the base container.
$ pip list | egrep 'transformer|core|nvidia|megatron'
httpcore                   1.0.9
jupyter_core               5.8.1
nvidia-cudnn-frontend      1.12.0
nvidia-dali-cuda120        1.50.0
nvidia-ml-py               12.575.51
nvidia-modelopt            0.29.0
nvidia-modelopt-core       0.29.0
nvidia-nvcomp-cu12         4.2.0.14
nvidia-nvimgcodec-cu12     0.5.0.13
nvidia-nvjpeg-cu12         12.4.0.16
nvidia-nvjpeg2k-cu12       0.8.1.40
nvidia-nvtiff-cu12         0.5.0.67
nvidia-resiliency-ext      0.4.0
pydantic_core              2.33.2
transformer_engine         2.4.0+3cd6870
```

### 2.1. Build TE wheel

First thing first: always, I repeat, always check the dependencies of the specific TE commit. You'll
need to get this correct. Below the snapshot of v2.7 (stable) and the upcoming v2.8. As you can see,
the next TE version has a new dependency `nvidia-mathdx==25.1.1`.

<https://github.com/NVIDIA/TransformerEngine/blob/release_v2.7/pyproject.toml#L6>

```python
["setuptools>=61.0", "cmake>=3.21", "wheel", "pybind11[global]", "ninja", "pip", "torch>=2.1", "jax[cuda12]", "flax>=0.7.1"]
```

<https://github.com/NVIDIA/TransformerEngine/blob/release_v2.8/pyproject.toml#L6>

```python
["setuptools>=61.0", "cmake>=3.21", "wheel", "pybind11[global]", "ninja", "nvidia-mathdx==25.1.1", "pip", "torch>=2.1", "jax>=0.5.0", "flax>=0.7.1"]
```

Now, let's build our TE wheel.

```bash
cd /opt/TransformerEngine

# This is TOT
git rev-parse --short HEAD ; git rev-parse HEAD; git show -s --format='%an%n%ad%n%s'
# 7e45be73
# 7e45be73bb8d513abe8785ee078ac88719bcd9f1
# Przemyslaw Tredak
# Sun Oct 5 16:48:27 2025 -0700
# Added the NVFP4 section to the low precision training tutorial (#2237)

pip install --no-build-isolation "nvidia-mathdx==25.1.1"

# To shorten build time, target just H100 and A1000 (my laptop).
# On DGX (8x H100, hundreds of CPU cores), the build time was ~5 minutes.
# On my laptop, it's probably hours (must set MAX_JOBS=1 to avoid OOM).
NVTE_CUDA_ARCHS="86;90" NVTE_FRAMEWORK=pytorch python setup.py bdist_wheel -v
```

After this, you should see the `.whl` file.

```console
$ find . -name '*.whl'
./dist/transformer_engine-2.9.0.dev0+7e45be73-cp312-cp312-linux_x86_64.whl

$ ls -al dist/*
-rw-r--r-- 1 root root 125641668 Oct  6 01:59 dist/transformer_engine-2.9.0.dev0+7e45be73-cp312-cp312-linux_x86_64.whl
```

To verify the TE installation now before installing MB, install the `.whl` (below), then refer to
section [2.3](#23-verify-te).

```bash
pip uninstall transformer_engine transformer_engine_torch transformer_engine_cu12

pip install -vvv '/opt/TransformerEngine/dist/transformer_engine-2.9.0.dev0+7e45be73-cp312-cp312-linux_x86_64.whl[pytorch]'
pip list | grep transformer engine
# transformer_engine         2.9.0.dev0+7e45be73
```

However, beware that that installing MBridge (next section) may downgrade the `transformer_engine`
back to 2.7, hence you'll need to reinstall this `.whl` again post MB install.

### 2.2. Install MB

```bash
cd /opt/Megatron-Bridge
git rev-parse --short HEAD ; git rev-parse HEAD ; git show -s --format='%an%n%ad%n%s'
# b30bded5
# b30bded5da5cf79d4f0864f609a06b9bea79d80b
# Oliver Koenig
# Mon Oct 6 06:16:45 2025 +0000
# beep boop 🤖: Bumping to v0.2.0rc7

# To allow later steps to update these packages
pip uninstall -y nvidia-modelopt nvidia-modelopt-core

# Through trial and error, identified the latest compatible triton with pytorch provided in NGC 25.06
pip install --no-cache-dir --no-build-isolation 'triton<3.4.0'
# ...
# Successfully installed triton-3.3.1

# Must match with Megatron-Bridge/pyproject.toml of the commit id.
# Example: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/762354696d0435f54b91baf37610a6d07e6fc75c/pyproject.toml#L68-81
#
# NOTE: make mandatory changes to mcore minimum version.
declare -a dependencies=(
    "datasets"
    "omegaconf>=2.3.0"
    "tensorboard>=2.19.0"
    "typing-extensions"
    "rich"
    "wandb>=0.19.10"
    "six>=1.17.0"
    "regex>=2024.11.6"
    "pyyaml>=6.0.2"
    "tqdm>=4.67.1"
    "hydra-core>1.3,<=1.3.2"

     # This pulls in causal-conv1d, mamba-ssm, nv_gruped_gemm.
     # Bump min version to provide MegatronLegacyTokenizer.
    "megatron-core[dev,mlm]>=0.15.0rc3,<0.16.0"
)
pip install "${dependencies[@]}"
# pip install --no-cache-dir "${dependencies[@]}"

export PYTHONPATH=/opt/Megatron-Bridge/src:$PYTHONPATH
echo PYTHONPATH=/opt/Megatron-Bridge/src:$PYTHONPATH >> /etc/environment
# pip install --no-build-isolation -e .

# Downgrade transformer to modelopt's requirement, to avoid runtime warning.
pip install 'nvidia-modelopt[hf]'

# Megatron core pins transformer_engine. Always check what your version or commit hash needs.
# Example: https://github.com/NVIDIA/Megatron-LM/blob/53008b844f98886a2144c216ecd25952cb2dda58/pyproject.toml#L69-L87
# E.g., 0.15.0rc7 (6-Oct-2025) pins to "transformer_engine>=2.6.0a0,<2.8.0".
#
# This means, to epxeriment with newer TE, after installing mbridge (and its deps), uninstall TE,
# then install own-build TE.
pip list | grep 'transformer.engine.*'
pip uninstall -y transformer-engine transformer-engine-torch transformer-engine-cu12
pip install '/opt/TransformerEngine/dist/transformer_engine-2.9.0.dev0+7e45be73-cp312-cp312-linux_x86_64.whl[pytorch]'
```

### 2.3. Verify TE

```bash
# https://github.com/NVIDIA/TransformerEngine?tab=readme-ov-file#pytorch
python -c 'import transformer_engine.pytorch as te ; print(te.__path__)'
python -c '
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048

# Initialize model and inputs.
model = te.Linear(in_features, out_features, bias=True)
inp = torch.randn(hidden_size, in_features, device="cuda")

# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

# Enable autocasting for the forward pass
for _ in range(100_000):   # HAHA: give me enough time to `nvidia-smi -l 1` in another terminal
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model(inp)

loss = out.sum()
loss.backward()
'

# On GPU without fp8 capability.
python -c '
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048

# Initialize model and inputs.
model = te.Linear(in_features, out_features, bias=True)
inp = torch.randn(hidden_size, in_features, device="cuda")

out = model(inp)
loss = out.sum()
loss.backward()
'
```

### 2.4. Verify MB

WARNING: MBridge API evolves rapidly. In the span of just one day, there's already a reorg on the
recipe.

First, simple imports.

```bash
python -c " from megatron.bridge import AutoBridge

from megatron.bridge.recipes.llama import llama32_1b_pretrain_config from
megatron.bridge.training.gpt_step import forward_step from megatron.bridge.training.pretrain import
pretrain
"
```

Next, train a small model. Needs HF credentials in case you haven't cached the model checkpoints and
the tokenizer.

```bash
# huggingface-cli is deprecated. Use hf ... as recommended by HF.
hf auth login login
hf auth whoami
```

Lastly, run a training. NOTE: this is also the test that can uncover PT <> Triton incompatibilities.

```bash
cat << 'EOF' > /workspace/train.py
from megatron.bridge import AutoBridge

from megatron.bridge.recipes.llama import llama32_1b_pretrain_config from
megatron.bridge.training.gpt_step import forward_step from megatron.bridge.training.pretrain import
pretrain

if __name__ == "__main__": # Load Llama from Hugging Face Hub and convert to Megatron bridge =
    AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B") model_provider =
    bridge.to_megatron_provider()

    seq_length = 1024
    #seq_length = 131072   # Match the seq_length in model provider. WARNING: needs TP=8, and even then, OOM even with ~1.9TB RAM

    # Get defaults for other configuration from an existing Llama 3.2 recipe
    # HAHA. WAR. Avoid failing assert lr_wamup_steps < lr_decay_steps
    cfg = llama32_1b_pretrain_config(mock=True, seq_length=seq_length, lr_decay_iters=10000, tensor_parallelism=8)
    cfg.model = model_provider

    # https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/879
    #
    # Reset back cfg.model.seq_length to the one I want.
    # model_provider.seq_length=131072 is not runnable on 8x H100.
    # Setting TP=8 can progress until data loader, but hits CPU OOM (RAM=1.9TB, num_workers=1)
    # See model_provider.txt and model_cfg.txt on the diff between model_provder and model_cfg (recipe).
    cfg.model.seq_length = seq_length

    cfg.train.train_iters = 10

    cfg.dataset.seq_length = cfg.model.seq_length
    cfg.dataset.sequence_length = cfg.model.seq_length
    cfg.tokenizer.vocab_size = cfg.model.vocab_size

    cfg.dataset.num_workers=1

    pretrain(cfg, forward_step)
EOF

# NOTE: seq_length=1024 can run on tp=1. But tp=8 to speeds-up the script runtime, and the short
# wait time helps preventing my mind from getting distracted :P
torchrun --standalone --nproc_per_node=8 /workspace/train.py
```

## 3. Appendix: training log

I like to snapshot GPU utilizations during training from a separate terminal in the host. At the
very least, to get a rough estimate of per-GPU VRAM utilization, and get an idea of the GPU
occupancies for this small test case.

```console
# -l 2 means to print every two seconds
$ nvidia-smi -l 2

Mon Oct  6 03:29:21 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 80GB HBM3          On  | 00000000:1B:00.0 Off |                    0 |
| N/A   36C    P0             259W / 700W |  18449MiB / 81559MiB |     25%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA H100 80GB HBM3          On  | 00000000:43:00.0 Off |                    0 |
| N/A   36C    P0             264W / 700W |  16273MiB / 81559MiB |     25%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA H100 80GB HBM3          On  | 00000000:52:00.0 Off |                    0 |
| N/A   36C    P0             266W / 700W |  16273MiB / 81559MiB |     27%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA H100 80GB HBM3          On  | 00000000:61:00.0 Off |                    0 |
| N/A   35C    P0             268W / 700W |  16273MiB / 81559MiB |     27%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   4  NVIDIA H100 80GB HBM3          On  | 00000000:9D:00.0 Off |                    0 |
| N/A   36C    P0             246W / 700W |  16273MiB / 81559MiB |     26%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   5  NVIDIA H100 80GB HBM3          On  | 00000000:C3:00.0 Off |                    0 |
| N/A   36C    P0             249W / 700W |  16273MiB / 81559MiB |     24%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   6  NVIDIA H100 80GB HBM3          On  | 00000000:D1:00.0 Off |                    0 |
| N/A   35C    P0             263W / 700W |  16273MiB / 81559MiB |     28%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   7  NVIDIA H100 80GB HBM3          On  | 00000000:DF:00.0 Off |                    0 |
| N/A   34C    P0             245W / 700W |  16273MiB / 81559MiB |     24%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   3375095      C   /usr/bin/python                           18430MiB |
|    1   N/A  N/A   3375096      C   /usr/bin/python                           16254MiB |
|    2   N/A  N/A   3375097      C   /usr/bin/python                           16254MiB |
|    3   N/A  N/A   3375098      C   /usr/bin/python                           16254MiB |
|    4   N/A  N/A   3375099      C   /usr/bin/python                           16254MiB |
|    5   N/A  N/A   3375100      C   /usr/bin/python                           16254MiB |
|    6   N/A  N/A   3375101      C   /usr/bin/python                           16254MiB |
|    7   N/A  N/A   3375102      C   /usr/bin/python                           16254MiB |
+---------------------------------------------------------------------------------------+
```

And here comes the training log.

```console
root@eos0506:/opt/Megatron-Bridge# torchrun --standalone --nproc_per_node=8 /tmp/train.py
W1006 03:25:41.706000 3374946 torch/distributed/run.py:766]
W1006 03:25:41.706000 3374946 torch/distributed/run.py:766] *****************************************
W1006 03:25:41.706000 3374946 torch/distributed/run.py:766] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W1006 03:25:41.706000 3374946 torch/distributed/run.py:766] *****************************************
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
/tmp/train.py:28: UserWarning: Function 'pretrain' is experimental. APIs in this module are subject to change without notice.
  pretrain(cfg, forward_step)
/tmp/train.py:28: UserWarning: Function 'pretrain' is experimental. APIs in this module are subject to change without notice.
  pretrain(cfg, forward_step)
/tmp/train.py:28: UserWarning: Function 'pretrain' is experimental. APIs in this module are subject to change without notice.
  pretrain(cfg, forward_step)
INFO:megatron.core.num_microbatches_calculator:setting number of microbatches to constant 64
> initializing torch distributed ...
/tmp/train.py:28: UserWarning: Function 'pretrain' is experimental. APIs in this module are subject to change without notice.
  pretrain(cfg, forward_step)
/tmp/train.py:28: UserWarning: Function 'pretrain' is experimental. APIs in this module are subject to change without notice.
  pretrain(cfg, forward_step)
/tmp/train.py:28: UserWarning: Function 'pretrain' is experimental. APIs in this module are subject to change without notice.
  pretrain(cfg, forward_step)
/tmp/train.py:28: UserWarning: Function 'pretrain' is experimental. APIs in this module are subject to change without notice.
  pretrain(cfg, forward_step)
/tmp/train.py:28: UserWarning: Function 'pretrain' is experimental. APIs in this module are subject to change without notice.
  pretrain(cfg, forward_step)
[Gloo] Rank 0 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 1 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 2 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 3 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 4 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 5 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 6 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 7 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 6 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 0 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 1 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank [Gloo] Rank 23 is connected to  is connected to 77 peer ranks.  peer ranks. Expected number of connected peer ranks is : Expected number of connected peer ranks is : 77

[Gloo] Rank 4 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 5 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 7 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 1 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 0 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 2 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank 5 is connected to 7 peer ranks. Expected number of connected peer ranks is : 7
[Gloo] Rank [Gloo] Rank 43 is connected to  is connected to 77 peer ranks.  peer ranks. Expected number of connected peer ranks is : Expected number of connected peer ranks is : 77

[Gloo] Rank [Gloo] Rank 67 is connected to  is connected to 77 peer ranks.  peer ranks. Expected number of connected peer ranks is : Expected number of connected peer ranks is : 77

> initialized tensor model parallel with size 1
> initialized pipeline model parallel with size 1
> setting random seeds to 1234 ...
time to initialize megatron (seconds): 8.337
[after megatron is initialized] datetime: 2025-10-06 03:26:12
> building NullTokenizer tokenizer ...
[after tokenizer is built] datetime: 2025-10-06 03:26:12
/usr/local/lib/python3.12/dist-packages/megatron/core/models/gpt/gpt_layer_specs.py:103: UserWarning: The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated and will be removed soon. Please update your code accordingly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/megatron/core/models/gpt/gpt_layer_specs.py:103: UserWarning: The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated and will be removed soon. Please update your code accordingly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/megatron/core/models/gpt/gpt_layer_specs.py:103: UserWarning: The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated and will be removed soon. Please update your code accordingly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/megatron/core/models/gpt/gpt_layer_specs.py:103: UserWarning: The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated and will be removed soon. Please update your code accordingly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/megatron/core/models/gpt/gpt_layer_specs.py:103: UserWarning: The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated and will be removed soon. Please update your code accordingly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/megatron/core/models/gpt/gpt_layer_specs.py:103: UserWarning: The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated and will be removed soon. Please update your code accordingly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/megatron/core/models/gpt/gpt_layer_specs.py:103: UserWarning: The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated and will be removed soon. Please update your code accordingly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/megatron/core/models/gpt/gpt_layer_specs.py:103: UserWarning: The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated and will be removed soon. Please update your code accordingly.
  warnings.warn(
INFO:megatron.bridge.models.llama.llama_provider:Apply rope scaling with factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192.
INFO:megatron.bridge.models.llama.llama_provider:Apply rope scaling with factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192.
INFO:megatron.bridge.models.llama.llama_provider:Apply rope scaling with factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192.
INFO:megatron.bridge.models.llama.llama_provider:Apply rope scaling with factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192.
INFO:megatron.bridge.models.llama.llama_provider:Apply rope scaling with factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192.
INFO:megatron.bridge.models.llama.llama_provider:Apply rope scaling with factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192.
INFO:megatron.bridge.models.llama.llama_provider:Apply rope scaling with factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192.
INFO:megatron.bridge.models.llama.llama_provider:Apply rope scaling with factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192.
Fetching 1 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1487.87it/s]
Fetching 1 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 3093.14it/s]
Fetching 1 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1135.13it/s]
Fetching 1 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2310.91it/s]
Fetching 1 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2487.72it/s]
Fetching 1 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2928.98it/s]
Fetching 1 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2531.26it/s]
Fetching 1 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2610.02it/s]
Loading from meta-llama/Llama-3.2-1B ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 (98/98) LlamaBridge
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 1235814400
INFO:megatron.core.distributed.distributed_data_parallel:Setting up DistributedDataParallel with config DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=True, overlap_param_gather=True, align_param_gather=False, use_distributed_optimizer=True, num_distributed_optimizer_instances=1, check_for_nan_in_grad=True, check_for_large_grads=False, bucket_size=40000000, pad_buckets_for_high_nccl_busbw=False, average_in_collective=True, fp8_param_gather=False, reuse_grad_buf_for_mxfp8_param_ag=False, use_megatron_fsdp=False, use_custom_fsdp=False, data_parallel_sharding_strategy='no_shard', gradient_reduce_div_fusion=True, suggested_communication_unit_size=None, preserve_fp32_weights=True, keep_fp8_transpose_cache=False, nccl_ub=False, fsdp_double_buffer=False, outer_dp_sharding_strategy='no_shard', disable_symmetric_registration=False, delay_wgrad_compute=False)
INFO:megatron.core.distributed.param_and_grad_buffer:Number of buckets for gradient all-reduce / reduce-scatter: 17
Params for bucket 1 (50333696 elements, 50333696 padded size):
        module.decoder.final_layernorm.weight
        module.decoder.layers.15.mlp.linear_fc1.weight
        module.decoder.layers.15.mlp.linear_fc2.weight
Params for bucket 2 (60821504 elements, 60821504 padded size):
        module.decoder.layers.15.self_attention.linear_qkv.weight
        module.decoder.layers.15.self_attention.linear_proj.weight
        module.decoder.layers.14.mlp.linear_fc2.weight
        module.decoder.layers.15.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.15.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.14.mlp.linear_fc1.weight
Params for bucket 3 (60821504 elements, 60821504 padded size):
        module.decoder.layers.13.mlp.linear_fc2.weight
        module.decoder.layers.14.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.14.self_attention.linear_proj.weight
        module.decoder.layers.14.self_attention.linear_qkv.weight
        module.decoder.layers.14.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.13.mlp.linear_fc1.weight
Params for bucket 4 (60821504 elements, 60821504 padded size):
        module.decoder.layers.13.self_attention.linear_qkv.weight
        module.decoder.layers.13.self_attention.linear_proj.weight
        module.decoder.layers.12.mlp.linear_fc2.weight
        module.decoder.layers.13.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.13.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.12.mlp.linear_fc1.weight
Params for bucket 5 (60821504 elements, 60821504 padded size):
        module.decoder.layers.12.self_attention.linear_qkv.weight
        module.decoder.layers.12.self_attention.linear_proj.weight
        module.decoder.layers.11.mlp.linear_fc2.weight
        module.decoder.layers.12.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.12.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.11.mlp.linear_fc1.weight
Params for bucket 6 (60821504 elements, 60821504 padded size):
        module.decoder.layers.11.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.10.mlp.linear_fc2.weight
        module.decoder.layers.11.self_attention.linear_qkv.weight
        module.decoder.layers.11.self_attention.linear_proj.weight
        module.decoder.layers.11.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.10.mlp.linear_fc1.weight
Params for bucket 7 (60821504 elements, 60821504 padded size):
        module.decoder.layers.10.self_attention.linear_qkv.weight
        module.decoder.layers.10.self_attention.linear_proj.weight
        module.decoder.layers.9.mlp.linear_fc2.weight
        module.decoder.layers.10.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.10.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.9.mlp.linear_fc1.weight
Params for bucket 8 (60821504 elements, 60821504 padded size):
        module.decoder.layers.9.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.8.mlp.linear_fc2.weight
        module.decoder.layers.9.self_attention.linear_qkv.weight
        module.decoder.layers.9.self_attention.linear_proj.weight
        module.decoder.layers.9.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.8.mlp.linear_fc1.weight
Params for bucket 9 (60821504 elements, 60821504 padded size):
        module.decoder.layers.8.self_attention.linear_qkv.weight
        module.decoder.layers.8.self_attention.linear_proj.weight
        module.decoder.layers.7.mlp.linear_fc2.weight
        module.decoder.layers.8.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.8.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.7.mlp.linear_fc1.weight
Params for bucket 10 (60821504 elements, 60821504 padded size):
        module.decoder.layers.7.self_attention.linear_proj.weight
        module.decoder.layers.6.mlp.linear_fc2.weight
        module.decoder.layers.7.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.7.self_attention.linear_qkv.weight
        module.decoder.layers.7.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.6.mlp.linear_fc1.weight
Params for bucket 11 (60821504 elements, 60821504 padded size):
        module.decoder.layers.6.self_attention.linear_proj.weight
        module.decoder.layers.6.self_attention.linear_qkv.weight
        module.decoder.layers.6.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.6.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.5.mlp.linear_fc2.weight
        module.decoder.layers.5.mlp.linear_fc1.weight
Params for bucket 12 (60821504 elements, 60821504 padded size):
        module.decoder.layers.4.mlp.linear_fc2.weight
        module.decoder.layers.5.self_attention.linear_proj.weight
        module.decoder.layers.5.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.5.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.5.self_attention.linear_qkv.weight
        module.decoder.layers.4.mlp.linear_fc1.weight
Params for bucket 13 (60821504 elements, 60821504 padded size):
        module.decoder.layers.3.mlp.linear_fc2.weight
        module.decoder.layers.4.self_attention.linear_qkv.weight
        module.decoder.layers.4.self_attention.linear_proj.weight
        module.decoder.layers.4.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.4.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.3.mlp.linear_fc1.weight
Params for bucket 14 (60821504 elements, 60821504 padded size):
        module.decoder.layers.2.mlp.linear_fc2.weight
        module.decoder.layers.3.self_attention.linear_qkv.weight
        module.decoder.layers.3.self_attention.linear_proj.weight
        module.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.3.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.2.mlp.linear_fc1.weight
Params for bucket 15 (60821504 elements, 60821504 padded size):
        module.decoder.layers.2.self_attention.linear_qkv.weight
        module.decoder.layers.2.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.1.mlp.linear_fc2.weight
        module.decoder.layers.1.mlp.linear_fc1.weight
        module.decoder.layers.2.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.2.self_attention.linear_proj.weight
Params for bucket 16 (60821504 elements, 60821504 padded size):
        module.decoder.layers.1.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.0.mlp.linear_fc2.weight
        module.decoder.layers.1.self_attention.linear_qkv.weight
        module.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.1.self_attention.linear_proj.weight
        module.decoder.layers.0.mlp.linear_fc1.weight
Params for bucket 17 (273158144 elements, 273158144 padded size):
        module.decoder.layers.0.mlp.linear_fc1.layer_norm_weight
        module.embedding.word_embeddings.weight
        module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.0.self_attention.linear_qkv.weight
        module.decoder.layers.0.self_attention.linear_proj.weight
INFO:megatron.core.optimizer:Setting up optimizer with config OptimizerConfig(optimizer='adam', lr=0.0003, min_lr=3e-05, decoupled_lr=None, decoupled_min_lr=None, weight_decay=0.1, fp8_recipe='tensorwise', fp16=False, bf16=True, reuse_grad_buf_for_mxfp8_param_ag=False, params_dtype=torch.bfloat16, use_precision_aware_optimizer=False, store_param_remainders=True, main_grads_dtype=torch.float32, main_params_dtype=torch.float32, exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-05, decoupled_weight_decay=True, sgd_momentum=0.9, use_distributed_optimizer=True, overlap_param_gather=False, overlap_param_gather_with_optimizer_step=False, optimizer_cpu_offload=False, optimizer_offload_fraction=0.0, use_torch_optimizer_for_cpu_offload=False, overlap_cpu_optimizer_d2h_h2d=False, pin_cpu_grads=True, pin_cpu_params=True, clip_grad=1.0, log_num_zeros_in_grad=False, barrier_with_L1_time=False, timers=<megatron.core.timers.Timers object at 0x7efccf363c80>, config_logger_dir='')
INFO:megatron.core.optimizer_param_scheduler:> learning rate decay style: cosine
[after model, optimizer, and learning rate scheduler are built] datetime: 2025-10-06 03:26:20
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      5120
    validation: 16384
    test:       16384
> building train, validation, and test datasets for GPT ...
INFO:megatron.core.datasets.blended_megatron_dataset_builder:Building MockGPTDataset splits with sizes=(5120, 16384, 16384) and config=GPTDatasetConfig(dataloader_type='single', num_workers=1, data_sharding=True, pin_memory=True, persistent_workers=False, random_seed=1234, sequence_length=1024, blend=None, blend_per_split=None, multiple_validation_sets=None, full_validation=None, split='1,1,1', split_matrix=[(0, 0.3333333333333333), (0.3333333333333333, 0.6666666666666666), (0.6666666666666666, 1.0)], num_dataset_builder_threads=1, path_to_cache=None, mmap_bin_files=True, mock=True, tokenizer=<megatron.bridge.training.tokenizers.tokenizer._NullTokenizer object at 0x7efccef421b0>, mid_level_dataset_surplus=0.005, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False, create_attention_mask=True, drop_last_partial_validation_sequence=True, add_extra_token_to_sequence=True, object_storage_cache_path=None, skip_getting_attention_mask_from_dataset=True)
INFO:megatron.core.datasets.gpt_dataset:Build and save the MockGPTDataset train indices
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 66592
INFO:megatron.core.datasets.gpt_dataset:> total number of epochs: 1
INFO:megatron.core.datasets.gpt_dataset:Build and save the MockGPTDataset valid indices
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 66562
INFO:megatron.core.datasets.gpt_dataset:> total number of epochs: 1
INFO:megatron.core.datasets.gpt_dataset:Build and save the MockGPTDataset test indices
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 66686
INFO:megatron.core.datasets.gpt_dataset:> total number of epochs: 1
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2025-10-06 03:26:20
done with setup ...
(min, max) time across ranks (ms):
    model-and-optimizer-setup ......................: (8274.77, 8274.90)
    train/valid/test-data-iterators-setup ..........: (67.42, 80.35)
------- Task Configuration -------
_target_: megatron.bridge.training.config.ConfigContainer
checkpoint:
  _target_: megatron.bridge.training.config.CheckpointConfig
  async_save: false
  ckpt_assume_constant_structure: false
  ckpt_convert_format: null
  ckpt_convert_save: null
  ckpt_format: torch_dist
  ckpt_step: null
  dist_ckpt_strictness: assume_ok_unexpected
  exit_on_missing_checkpoint: false
  finetune: false
  fully_parallel_load: false
  fully_parallel_save: true
  load: /opt/Megatron-Bridge/nemo_experiments/default/checkpoints
  load_main_params_from_ckpt: false
  load_optim: true
  load_rng: true
  most_recent_k: -1
  non_persistent_ckpt_type: null
  non_persistent_global_ckpt_dir: null
  non_persistent_local_ckpt_algo: fully_parallel
  non_persistent_local_ckpt_dir: null
  non_persistent_save_interval: null
  pretrained_checkpoint: null
  replication: false
  replication_factor: 2
  replication_jump: null
  save: /opt/Megatron-Bridge/nemo_experiments/default/checkpoints
  save_interval: 500
  save_optim: true
  save_rng: true
  strict_fsdp_dtensor_load: false
  use_checkpoint_args: false
  use_persistent_ckpt_worker: true
comm_overlap: null
dataset:
  _target_: megatron.bridge.training.config.GPTDatasetConfig
  add_extra_token_to_sequence: true
  blend: null
  blend_per_split: null
  create_attention_mask: true
  data_sharding: true
  dataloader_type: single
  drop_last_partial_validation_sequence: true
  eod_mask_loss: false
  full_validation: null
  mid_level_dataset_surplus: 0.005
  mmap_bin_files: true
  mock: true
  multiple_validation_sets: null
  num_dataset_builder_threads: 1
  num_workers: 1
  object_storage_cache_path: null
  path_to_cache: null
  persistent_workers: false
  pin_memory: true
  random_seed: 1234
  reset_attention_mask: false
  reset_position_ids: false
  sequence_length: 1024
  skip_getting_attention_mask_from_dataset: true
  split: 1,1,1
  split_matrix:
  - - 0
    - 0.3333333333333333
  - - 0.3333333333333333
    - 0.6666666666666666
  - - 0.6666666666666666
    - 1.0
  tokenizer:
    _call_: true
    _target_: megatron.bridge.training.tokenizers.tokenizer._NullTokenizer
ddp:
  _target_: megatron.bridge.training.config.DistributedDataParallelConfig
  align_param_gather: false
  average_in_collective: true
  bucket_size: 40000000
  check_for_large_grads: false
  check_for_nan_in_grad: true
  data_parallel_sharding_strategy: no_shard
  delay_wgrad_compute: false
  disable_symmetric_registration: false
  fp8_param_gather: false
  fsdp_double_buffer: false
  grad_reduce_in_fp32: true
  gradient_reduce_div_fusion: true
  keep_fp8_transpose_cache: false
  nccl_ub: false
  num_distributed_optimizer_instances: 1
  outer_dp_sharding_strategy: no_shard
  overlap_grad_reduce: true
  overlap_param_gather: true
  pad_buckets_for_high_nccl_busbw: false
  preserve_fp32_weights: true
  reuse_grad_buf_for_mxfp8_param_ag: false
  suggested_communication_unit_size: null
  use_custom_fsdp: false
  use_distributed_optimizer: true
  use_megatron_fsdp: false
dist:
  _target_: megatron.bridge.training.config.DistributedInitConfig
  align_grad_reduce: true
  distributed_backend: nccl
  distributed_timeout_minutes: 10
  enable_megatron_core_experimental: false
  external_gpu_device_mapping: false
  high_priority_stream_groups: null
  lazy_init: false
  local_rank: 0
  nccl_communicator_config_path: null
  sharp_enabled_group: null
  use_gloo_process_groups: true
  use_megatron_fsdp: false
  use_sharp: false
  use_torch_fsdp2: false
  use_tp_pp_dp_mapping: false
ft: null
inprocess_restart: null
logger:
  _target_: megatron.bridge.training.config.LoggerConfig
  filter_warnings: true
  log_energy: false
  log_interval: 10
  log_loss_scale_to_tensorboard: true
  log_memory_to_tensorboard: false
  log_params_norm: false
  log_progress: false
  log_throughput: false
  log_timers_to_tensorboard: true
  log_validation_ppl_to_tensorboard: false
  log_world_size_to_tensorboard: false
  logging_level: 20
  modules_to_filter: null
  set_level_for_all_loggers: false
  tensorboard_dir: /opt/Megatron-Bridge/nemo_experiments/default/tb_logs
  tensorboard_log_interval: 1
  tensorboard_queue_size: 1000
  timing_log_level: 0
  timing_log_option: minmax
  wandb_entity: null
  wandb_exp_name: null
  wandb_project: null
  wandb_save_dir: null
mixed_precision:
  _target_: megatron.bridge.training.mixed_precision.MixedPrecisionConfig
  autocast_dtype: null
  autocast_enabled: false
  bf16: true
  first_last_layers_bf16: false
  fp16: false
  fp32: false
  fp8: null
  fp8_amax_compute_algo: most_recent
  fp8_amax_history_len: 1
  fp8_dot_product_attention: false
  fp8_margin: 0
  fp8_multi_head_attention: false
  fp8_param: false
  fp8_param_gather: false
  fp8_recipe: tensorwise
  fp8_wgrad: true
  grad_reduce_in_fp32: true
  hysteresis: 2
  initial_loss_scale: 4294967296
  loss_scale: null
  loss_scale_window: 1000
  min_loss_scale: 1.0
  num_layers_at_end_in_bf16: 0
  num_layers_at_start_in_bf16: 0
  params_dtype:
    _call_: false
    _target_: torch.bfloat16
  pipeline_dtype:
    _call_: false
    _target_: torch.bfloat16
  reuse_grad_buf_for_mxfp8_param_ag: false
model:
  _target_: megatron.bridge.models.llama.llama_provider.Llama31ModelProvider
  account_for_embedding_in_pipeline_split: false
  account_for_loss_in_pipeline_split: false
  activation_func:
    _call_: false
    _target_: torch.nn.functional.silu
  activation_func_clamp_value: null
  activation_func_fp8_input_store: false
  add_bias_linear: false
  add_qkv_bias: false
  apply_query_key_layer_scaling: false
  apply_residual_connection_post_layernorm: false
  apply_rope_fusion: true
  async_tensor_model_parallel_allreduce: false
  attention_backend:
    _args_:
    - 5
    _call_: true
    _target_: megatron.core.transformer.enums.AttnBackend
  attention_dropout: 0.0
  attention_softmax_in_fp32: false
  autocast_dtype:
    _call_: false
    _target_: torch.bfloat16
  barrier_with_L1_time: true
  batch_p2p_comm: true
  batch_p2p_sync: true
  bf16: true
  bias_activation_fusion: true
  bias_dropout_fusion: true
  calculate_per_token_loss: false
  clone_scatter_output_in_embedding: true
  config_logger_dir: ''
  context_parallel_size: 1
  cp_comm_type: null
  cpu_offloading: false
  cpu_offloading_activations: true
  cpu_offloading_double_buffering: false
  cpu_offloading_num_layers: 0
  cpu_offloading_weights: false
  cross_entropy_fusion_impl: native
  cross_entropy_loss_fusion: true
  cuda_graph_retain_backward_graph: false
  cuda_graph_scope: full
  cuda_graph_use_single_mempool: false
  cuda_graph_warmup_steps: 3
  deallocate_pipeline_outputs: true
  defer_embedding_wgrad_compute: false
  delay_wgrad_compute: false
  deterministic_mode: false
  disable_bf16_reduced_precision_matmul: false
  disable_parameter_transpose_cache: false
  distribute_saved_activations: null
  embedding_init_method:
    _args_: []
    _partial_: true
    _target_: torch.nn.init.normal_
    mean: 0.0
    std: 0.02
  embedding_init_method_std: 0.02
  enable_autocast: false
  enable_cuda_graph: false
  expert_model_parallel_size: 1
  expert_tensor_parallel_size: 1
  external_cuda_graph: false
  ffn_hidden_size: 8192
  finalize_model_grads_func:
    _call_: false
    _target_: megatron.core.distributed.finalize_model_grads.finalize_model_grads
  first_last_layers_bf16: false
  flash_decode: false
  fp16: false
  fp16_lm_cross_entropy: false
  fp32_residual_connection: false
  fp4: null
  fp4_param: false
  fp4_recipe: nvfp4
  fp8: null
  fp8_amax_compute_algo: most_recent
  fp8_amax_history_len: 1
  fp8_dot_product_attention: false
  fp8_interval: 1
  fp8_margin: 0
  fp8_multi_head_attention: false
  fp8_param: false
  fp8_recipe: tensorwise
  fp8_wgrad: true
  fused_single_qkv_rope: false
  gated_linear_unit: true
  generation_config:
    _call_: true
    _target_: transformers.generation.configuration_utils.GenerationConfig.from_dict
    config_dict:
      _from_model_config: true
      assistant_confidence_threshold: 0.4
      assistant_early_exit: null
      assistant_lookbehind: 10
      bad_words_ids: null
      begin_suppress_tokens: null
      bos_token_id: 128000
      cache_config: null
      cache_implementation: null
      constraints: null
      decoder_start_token_id: null
      disable_compile: false
      diversity_penalty: 0.0
      do_sample: true
      dola_layers: null
      early_stopping: false
      encoder_no_repeat_ngram_size: 0
      encoder_repetition_penalty: 1.0
      eos_token_id: 128001
      epsilon_cutoff: 0.0
      eta_cutoff: 0.0
      exponential_decay_length_penalty: null
      force_words_ids: null
      forced_bos_token_id: null
      forced_eos_token_id: null
      guidance_scale: null
      is_assistant: false
      length_penalty: 1.0
      low_memory: null
      max_length: 20
      max_matching_ngram_size: null
      max_new_tokens: null
      max_time: null
      min_length: 0
      min_new_tokens: null
      min_p: null
      no_repeat_ngram_size: 0
      num_assistant_tokens: 20
      num_assistant_tokens_schedule: constant
      num_beam_groups: 1
      num_beams: 1
      num_return_sequences: 1
      output_attentions: false
      output_hidden_states: false
      output_logits: null
      output_scores: false
      pad_token_id: null
      penalty_alpha: null
      prefill_chunk_size: null
      prompt_lookup_num_tokens: null
      remove_invalid_values: false
      renormalize_logits: false
      repetition_penalty: 1.0
      return_dict_in_generate: false
      return_legacy_cache: null
      sequence_bias: null
      stop_strings: null
      suppress_tokens: null
      target_lookbehind: 10
      temperature: 0.6
      token_healing: false
      top_k: 50
      top_p: 0.9
      transformers_version: 4.53.3
      typical_p: 1.0
      use_cache: true
      watermarking_config: null
  glu_linear_offset: 0.0
  grad_scale_func:
    _call_: false
    _target_: megatron.core.optimizer.optimizer.MegatronOptimizer.scale_loss
  grad_sync_func:
    _call_: false
    _target_: megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.start_grad_sync
  gradient_accumulation_fusion: true
  hetereogenous_dist_checkpoint: false
  heterogeneous_block_specs: false
  hidden_dropout: 0.0
  hidden_size: 2048
  hierarchical_context_parallel_sizes: null
  high_freq_factor: 4.0
  inference_rng_tracker: false
  inference_sampling_seed: 42
  init_method:
    _args_: []
    _partial_: true
    _target_: torch.nn.init.normal_
    mean: 0.0
    std: 0.02
  init_method_std: 0.02
  init_model_with_meta_device: false
  is_hybrid_model: false
  kv_channels: 64
  layernorm_epsilon: 1.0e-05
  layernorm_zero_centered_gamma: false
  low_freq_factor: 1.0
  make_vocab_size_divisible_by: 128
  mamba_head_dim: 64
  mamba_num_groups: 8
  mamba_num_heads: null
  mamba_state_dim: 128
  masked_softmax_fusion: true
  memory_efficient_layer_norm: false
  microbatch_group_size_per_vp_stage: 1
  mlp_chunks_for_prefill: 1
  moe_apply_probs_on_input: false
  moe_aux_loss_coeff: 0.0
  moe_deepep_num_sms: 20
  moe_enable_deepep: false
  moe_expert_capacity_factor: null
  moe_extended_tp: false
  moe_ffn_hidden_size: null
  moe_grouped_gemm: false
  moe_input_jitter_eps: null
  moe_layer_freq: 1
  moe_layer_recompute: false
  moe_pad_expert_input_to_capacity: false
  moe_per_layer_logging: false
  moe_permute_fusion: false
  moe_router_bias_update_rate: 0.001
  moe_router_dtype: null
  moe_router_enable_expert_bias: false
  moe_router_force_load_balancing: false
  moe_router_fusion: false
  moe_router_group_topk: null
  moe_router_load_balancing_type: aux_loss
  moe_router_num_groups: null
  moe_router_padding_for_fp8: false
  moe_router_pre_softmax: false
  moe_router_score_function: softmax
  moe_router_topk: 2
  moe_router_topk_limited_devices: null
  moe_router_topk_scaling_factor: null
  moe_shared_expert_intermediate_size: null
  moe_shared_expert_overlap: false
  moe_token_dispatcher_type: allgather
  moe_token_drop_policy: probs
  moe_token_dropping: false
  moe_use_legacy_grouped_gemm: false
  moe_z_loss_coeff: null
  mrope_section: null
  mtp_enabled: false
  mtp_loss_scaling_factor: null
  mtp_num_layers: null
  multi_latent_attention: false
  no_rope_freq: null
  no_sync_func:
    _call_: false
    _target_: megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.no_sync
  normalization: RMSNorm
  num_attention_heads: 32
  num_layers: 16
  num_layers_at_end_in_bf16: 0
  num_layers_at_start_in_bf16: 0
  num_layers_in_first_pipeline_stage: null
  num_layers_in_last_pipeline_stage: null
  num_microbatches_with_partial_activation_checkpoints: null
  num_moe_experts: null
  num_query_groups: 8
  old_context_len: 8192
  output_layer_init_method:
    _args_: []
    _partial_: true
    _target_: torch.nn.init.normal_
    mean: 0.0
    std: 0.0035355339059327372
  overlap_moe_expert_parallel_comm: false
  overlap_p2p_comm: false
  overlap_p2p_comm_warmup_flush: false
  parallel_output: true
  param_sync_func: null
  params_dtype:
    _call_: false
    _target_: torch.bfloat16
  perform_initialization: false
  persist_layer_norm: false
  pipeline_dtype:
    _call_: false
    _target_: torch.bfloat16
  pipeline_model_parallel_comm_backend: null
  pipeline_model_parallel_layout: null
  pipeline_model_parallel_size: 1
  position_embedding_type: rope
  qk_layernorm: false
  quant_recipe: null
  recompute_granularity: null
  recompute_method: null
  recompute_modules:
  - core_attn
  recompute_num_layers: null
  restore_modelopt_state: false
  rotary_base: 500000.0
  rotary_interleaved: false
  rotary_percent: 1.0
  scale_factor: 32.0
  scatter_embedding_sequence_parallel: true
  seq_len_interpolation_factor: null
  seq_length: 131072
  sequence_parallel: false
  share_embeddings_and_output_weights: true
  should_pad_vocab: false
  softmax_scale: null
  softmax_type: vanilla
  symmetric_ar_type: null
  tensor_model_parallel_size: 1
  test_mode: false
  timers:
    _call_: true
    _target_: megatron.core.timers.Timers
  tp_comm_atomic_ag: false
  tp_comm_atomic_rs: false
  tp_comm_bootstrap_backend: nccl
  tp_comm_bulk_dgrad: true
  tp_comm_bulk_wgrad: true
  tp_comm_overlap: false
  tp_comm_overlap_ag: true
  tp_comm_overlap_cfg: null
  tp_comm_overlap_disable_fc1: false
  tp_comm_overlap_disable_qkv: false
  tp_comm_overlap_rs: true
  tp_comm_overlap_rs_dgrad: false
  tp_comm_split_ag: true
  tp_comm_split_rs: true
  tp_only_amax_red: false
  transformer_impl: transformer_engine
  transformer_layer_spec:
    _call_: false
    _target_: megatron.bridge.models.gpt_provider.default_layer_spec
  use_cpu_initialization: false
  use_fused_weighted_squared_relu: false
  use_kitchen: false
  use_mamba_mem_eff_path: true
  use_ring_exchange_p2p: false
  use_te_activation_func: false
  use_te_rng_tracker: false
  use_transformer_engine_full_layer_spec: false
  use_transformer_engine_op_fuser: null
  variable_seq_lengths: false
  virtual_pipeline_model_parallel_size: null
  vocab_size: 128256
  wgrad_deferral_limit: 0
  window_attn_skip_freq: null
  window_size: null
nvrx_straggler: null
optimizer:
  _target_: megatron.bridge.training.config.OptimizerConfig
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1.0e-05
  barrier_with_L1_time: false
  bf16: true
  clip_grad: 1.0
  config_logger_dir: ''
  decoupled_lr: null
  decoupled_min_lr: null
  decoupled_weight_decay: true
  exp_avg_dtype:
    _call_: false
    _target_: torch.float32
  exp_avg_sq_dtype:
    _call_: false
    _target_: torch.float32
  fp16: false
  fp8_recipe: tensorwise
  hysteresis: 2
  initial_loss_scale: 4294967296
  log_num_zeros_in_grad: false
  loss_scale: null
  loss_scale_window: 1000
  lr: 0.0003
  main_grads_dtype:
    _call_: false
    _target_: torch.float32
  main_params_dtype:
    _call_: false
    _target_: torch.float32
  min_loss_scale: 1.0
  min_lr: 3.0e-05
  optimizer: adam
  optimizer_cpu_offload: false
  optimizer_offload_fraction: 0.0
  overlap_cpu_optimizer_d2h_h2d: false
  overlap_param_gather: false
  overlap_param_gather_with_optimizer_step: false
  params_dtype:
    _call_: false
    _target_: torch.bfloat16
  pin_cpu_grads: true
  pin_cpu_params: true
  reuse_grad_buf_for_mxfp8_param_ag: false
  sgd_momentum: 0.9
  store_param_remainders: true
  timers:
    _call_: true
    _target_: megatron.core.timers.Timers
  use_distributed_optimizer: true
  use_precision_aware_optimizer: false
  use_torch_optimizer_for_cpu_offload: false
  weight_decay: 0.1
peft: null
profiling: null
rerun_state_machine:
  _target_: megatron.bridge.training.config.RerunStateMachineConfig
  check_for_nan_in_loss: true
  check_for_spiky_loss: false
  error_injection_rate: 0
  error_injection_type: transient_error
  rerun_mode: disabled
rng:
  _target_: megatron.bridge.training.config.RNGConfig
  data_parallel_random_init: false
  inference_rng_tracker: false
  seed: 1234
  te_rng_tracker: false
scheduler:
  _target_: megatron.bridge.training.config.SchedulerConfig
  end_weight_decay: 0.033
  lr_decay_iters: 10000
  lr_decay_steps: 5120000
  lr_decay_style: cosine
  lr_warmup_fraction: null
  lr_warmup_init: 0.0
  lr_warmup_iters: 2000
  lr_warmup_steps: 1024000
  lr_wsd_decay_iters: null
  lr_wsd_decay_style: exponential
  override_opt_param_scheduler: true
  start_weight_decay: 0.033
  use_checkpoint_opt_param_scheduler: false
  wd_incr_steps: 5120
  weight_decay_incr_style: constant
  wsd_decay_steps: null
straggler: null
tokenizer:
  _target_: megatron.bridge.training.tokenizers.config.TokenizerConfig
  image_tag_type: null
  merge_file: null
  special_tokens: null
  tiktoken_num_special_tokens: 1000
  tiktoken_pattern: null
  tiktoken_special_tokens: null
  tokenizer_model: null
  tokenizer_prompt_format: null
  tokenizer_type: NullTokenizer
  vocab_extra_ids: 0
  vocab_file: null
  vocab_size: 128256
train:
  _target_: megatron.bridge.training.config.TrainingConfig
  check_weight_hash_across_dp_replicas_interval: null
  decrease_batch_size_if_needed: false
  empty_unused_memory_level: 0
  eval_interval: 2000
  eval_iters: 32
  exit_duration_in_mins: null
  exit_interval: null
  exit_signal:
    _args_:
    - 15
    _call_: true
    _target_: signal.Signals
  exit_signal_handler: false
  exit_signal_handler_for_dataloader: false
  global_batch_size: 512
  manual_gc: true
  manual_gc_eval: 100
  manual_gc_interval: 100
  micro_batch_size: 1
  rampup_batch_size: null
  skip_train: false
  train_iters: 10
  train_sync_interval: null

----------------------------------
Training ...
Setting rerun_state_machine.current_iteration to 0...
> setting tensorboard ...
Step Time : 4.65s GPU utilization: 59892.5TFLOP/s/GPU
Number of parameters in transformer layers in billions:  0.97
Number of parameters in embedding layers in billions: 0.26
Total number of parameters in billions: 1.24
Number of parameters in most loaded shard in billions: 1.2359
Theoretical memory footprints: weight and optimizer=8839.72 MB
 [2025-10-06 03:27:07] iteration       10/      10 | consumed samples:         5120 | elapsed time per iteration (ms): 4647.8 | learning rate: 1.500000E-06 | global batch size:   512 | lm loss: 9.763827E+00 | loss scale: 1.0 | grad norm: 33.055 | number of skipped iterations:   0 | number of nan iterations:   0 |
[Rank 0] (after 10 iterations) memory (MB) | allocated: 9554.87939453125 | max allocated: 12003.60400390625 | reserved: 12232.0 | max reserved: 12232.0
[after training is done] datetime: 2025-10-06 03:27:07
saving checkpoint at iteration      10 to /opt/Megatron-Bridge/nemo_experiments/default/checkpoints in torch_dist format
Storing distributed optimizer sharded state of type fully_sharded_model_space
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
WARNING:megatron.core.utils:fused_indices_to_multihot has reached end of life. Please migrate to a non-experimental function.
  successfully saved checkpoint from iteration      10 to /opt/Megatron-Bridge/nemo_experiments/default/checkpoints [ t 1/1, p 1/1 ]
Evaluating on 16384 samples
Evaluating iter 1/32
Evaluating iter 2/32
Evaluating iter 3/32
Evaluating iter 4/32
Evaluating iter 5/32
Evaluating iter 6/32
Evaluating iter 7/32
Evaluating iter 8/32
Evaluating iter 9/32
Evaluating iter 10/32
Evaluating iter 11/32
Evaluating iter 12/32
Evaluating iter 13/32
Evaluating iter 14/32
Evaluating iter 15/32
Evaluating iter 16/32
Evaluating iter 17/32
Evaluating iter 18/32
Evaluating iter 19/32
Evaluating iter 20/32
Evaluating iter 21/32
Evaluating iter 22/32
Evaluating iter 23/32
Evaluating iter 24/32
Evaluating iter 25/32
Evaluating iter 26/32
Evaluating iter 27/32
Evaluating iter 28/32
Evaluating iter 29/32
Evaluating iter 30/32
Evaluating iter 31/32
Evaluating iter 32/32
(min, max) time across ranks (ms):
    evaluate .......................................: (57313.42, 57314.02)
----------------------------------------------------------------------------------------------------------------
 validation loss at iteration 10 on validation set | lm loss value: 9.647480E+00 | lm loss PPL: 1.548272E+04 |
----------------------------------------------------------------------------------------------------------------
Evaluating on 16384 samples
Evaluating iter 1/32
Evaluating iter 2/32
Evaluating iter 3/32
Evaluating iter 4/32
Evaluating iter 5/32
Evaluating iter 6/32
Evaluating iter 7/32
Evaluating iter 8/32
Evaluating iter 9/32
Evaluating iter 10/32
Evaluating iter 11/32
Evaluating iter 12/32
Evaluating iter 13/32
Evaluating iter 14/32
Evaluating iter 15/32
Evaluating iter 16/32
Evaluating iter 17/32
Evaluating iter 18/32
Evaluating iter 19/32
Evaluating iter 20/32
Evaluating iter 21/32
Evaluating iter 22/32
Evaluating iter 23/32
Evaluating iter 24/32
Evaluating iter 25/32
Evaluating iter 26/32
Evaluating iter 27/32
Evaluating iter 28/32
Evaluating iter 29/32
Evaluating iter 30/32
Evaluating iter 31/32
Evaluating iter 32/32
(min, max) time across ranks (ms):
    evaluate .......................................: (56212.02, 56212.65)
----------------------------------------------------------------------------------------------------------
 validation loss at iteration 10 on test set | lm loss value: 9.625707E+00 | lm loss PPL: 1.514926E+04 |
----------------------------------------------------------------------------------------------------------
[rank0]:[W1006 03:29:26.926188014 ProcessGroupNCCL.cpp:1505] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[rank2]:[W1006 03:29:27.782602114 ProcessGroupNCCL.cpp:1505] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[rank1]:[W1006 03:29:27.988342345 ProcessGroupNCCL.cpp:1505] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[rank5]:[W1006 03:29:27.047629126 ProcessGroupNCCL.cpp:1505] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[rank3]:[W1006 03:29:27.325919458 ProcessGroupNCCL.cpp:1505] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[rank7]:[W1006 03:29:28.838565410 ProcessGroupNCCL.cpp:1505] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[rank4]:[W1006 03:29:28.838748119 ProcessGroupNCCL.cpp:1505] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[rank6]:[W1006 03:29:28.918457984 ProcessGroupNCCL.cpp:1505] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```

## 4. Appendix: `nemo:25.09.00`

Probe versions provided by this NGC NeMo container.

```console
$ pip list | egrep 'transformer|core|nvidia|megatron'
botocore                           1.40.43
google-api-core                    2.25.1
httpcore                           1.0.9
hydra-core                         1.3.2
jupyter_core                       5.8.1
megatron-bridge                    0.1.0rc3                      /opt/Megatron-Bridge
megatron-core                      0.14.0rc8                     /opt/megatron-lm
megatron-energon                   5.2.0
megatron-fsdp                      0.1.0rc4
nvidia-cudnn-frontend              1.14.1
nvidia-dali-cuda120                1.50.0
nvidia-eval-commons                1.0.5
nvidia_lm_eval                     25.8.1
nvidia-ml-py                       13.580.82
nvidia-modelopt                    0.35.1                        /opt/TensorRT-Model-Optimizer
nvidia-modelopt-core               0.29.0
nvidia-nvcomp-cu12                 4.2.0.14
nvidia-nvimgcodec-cu12             0.5.0.13
nvidia-nvjitlink-cu12              12.8.93
nvidia-nvjpeg-cu12                 12.4.0.16
nvidia-nvjpeg2k-cu12               0.8.1.40
nvidia-nvtiff-cu12                 0.5.0.67
nvidia-nvtx-cu12                   12.8.90
nvidia-pytriton                    0.7.0
nvidia-resiliency-ext              0.4.1
nvidia-sphinx-theme                0.0.8
pyannote.core                      5.0.0
pydantic_core                      2.37.2
rouge_score                        0.1.2
sentence-transformers              5.1.1
taming-transformers                0.0.1
transformer_engine                 2.7.0.post0+c02ce232
transformers                       4.53.3
```
