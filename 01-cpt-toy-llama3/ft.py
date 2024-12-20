"""
Huiying Li <huiyingl@nvidia.com>

[This] script can 1. load tokenizer, 2. won't try to load optimizer state, and
can start a lora finetuning successfully. I tested it in 24.09 container.

Please review and change as necessary (e.g., change the hardcoded paths, etc.)

Usage: python ft.py
"""
import pytorch_lightning as pl
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from dataclasses import dataclass


def trainer(devices=1) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
    )

    return nl.Trainer(
        devices=1,
        max_steps=40,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        log_every_n_steps=1,
        limit_val_batches=2,
        val_check_interval=2,
        num_sanity_val_steps=0,
    )


def logger() -> nl.NeMoLogger:
    ckpt = nl.ModelCheckpoint(
        save_last=True,
        every_n_train_steps=10,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    return nl.NeMoLogger(
        name="nemo2_peft",
        log_dir="/tmp/exp",
        use_datetime_version=False,
        ckpt=ckpt,
        wandb=None,
    )


def adam_with_cosine_annealing() -> nl.OptimizerModule:
    return nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer="adam",
            lr=0.0001,
            adam_beta2=0.98,
            use_distributed_optimizer=True,
            clip_grad=1.0,
            bf16=True,
        ),
    )


def lora() -> nl.pytorch.callbacks.PEFT:
    return llm.peft.LoRA()


def squad() -> pl.LightningDataModule:
    return llm.SquadDataModule(seq_length=2048, micro_batch_size=2, global_batch_size=8, num_workers=0)

@dataclass
class Llama3ConfigToy(llm.Llama3Config8B):
    cross_entropy_loss_fusion: bool = False
    num_layers: int = 1
    hidden_size: int = 256
    num_attention_heads: int = 32
    ffn_hidden_size: int = 512
    seq_length: int = 128


def llama3_toy() -> pl.LightningModule:
    return llm.LlamaModel(Llama3ConfigToy())



# @run.factory
def resume() -> nl.AutoResume:
    return nl.AutoResume(
        restore_config=nl.RestoreConfig(
            path="/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last/",
        ),
        resume_if_exists=True,
    )


if __name__ == '__main__':
    llm.finetune(
        model=llama3_toy(),
        data=squad(),
        trainer=trainer(),
        peft=lora(),
        log=logger(),
        optim=adam_with_cosine_annealing(),
        resume=resume(),
    )
