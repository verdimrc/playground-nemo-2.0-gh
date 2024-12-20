# Continual pretraining (or continual sft)

The examples in this folder run on a local machine.

```bash
# For simplicity, we run the docker container as root (which is the default nemo:24.09).
docker run -it --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/haha \
    nvcr.io/nvidia/nemo:24.09 \
    /bin/bash

# Below are all under the container

# 01: generate a toy checkpoint without optimizer.
# The final checkpoint contains weights + tokenizer.
#
# Add --yes to skip the confirmation
/haha/01-pretrain.sh log.ckpt.save_optim_on_train_end=False

# 02a: continual pretraining using finetuning recipe (lora disabled). This
# version uses the nemo run CLI. This recipe uses SquadDataset (will download if
# not avail on your local HF cache).
#
# Add --yes to skip the confirmation
/haha/02-cpt-cli.sh

# 02b: continual pretraining using finetuning recipe (lora disabled). This
# version uses a custom recipe (Pythonic) to use MockDataset.
#
# Add --yes to skip the confirmation
/haha/02-cpt-custom-recipe-mockds.sh
```

## Appendix

1. [ft.py](ft.py) shows how to use Nemo-2.0 without recipe, and without nemo run. Instead, it
   directly uses the fine-grained construts such as model, trainer, etc.
2. `pretrain.log.gz` is an intentionally versioned sample log (but compressed, to save space).
   VScode users may use the
   [hyunkyunmoon.gzipdecompressor](https://github.com/hyeongyun0916/GZIP_Decompressor) extension to
   quickly view it with 1-2 clicks.
