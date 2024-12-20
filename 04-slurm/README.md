# Running Nemo-2.0 on Slurm

## Pre-requisites

1. Ensure Slurm cluster has enroot + pyxis. To validate, see the examples in
   <https://gist.github.com/verdimrc/d6f0c5f0f66debb5383f66ed9ac3da31#1-quickrun>
2. Decide where the Nemo client should be. It could be your laptop, or the Slurm login node.
3. Nemo client must be able to `docker pull`.
4. Nemo client must be able to ssh to the Slurm login (or head) node. As a best practice, setup the
   keypair authentication between these machines.

## On the Slurm login node

1. `docker pull nvcr.io/nvidia/nemo:24.09`
2. Convert the docker image to enroot `.sqsh` file. Make sure the `.sqsh` file sits on a shared fs
   to allow all nodes can read it.
3. Take note of all the shared fs paths.
4. Optional: ensure the dataset and initial checkpoints are already downloaded to the shared fs. You
   can do this as a one-time activity on the cluster.

## On the client machine

1. Get the [.py example in Nemo GH
   repo](https://github.com/NVIDIA/NeMo/blob/main/examples/llm/pretrain/default_executor.py).
2. `docker pull nvcr.io/nvidia/nemo:24.09`. Skip if the client machine is the Slurm login node.
3. Modify the example to use the `.sqsh` file, set hostname, etc. You may also modify the recipe
   into a tiny custom model. See the `*.py` examples. Please note that the
   container image path (and the input + output paths) are on the cluster (and should be the shared
   fs.)
4. Start a container, either `docker run`, or if you're on the Slurm login node, you may run the
   enroot container already created on the login node.
5. On the client container, run the example.

For step 4 and 5, you may refer to the steps outlined in `scratch*.sh`. A `*_xx.py` file should have
a matching `scratch-xx-*.sh` file.
