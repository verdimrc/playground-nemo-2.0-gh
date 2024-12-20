from pathlib import Path

## /opt/NeMo/nemo/collections/llm/api.py
from nemo.collections import llm

# Main guard needed, otherwise hang. Fix courtesy of @Chen Cui, @Ao Tang.
# https://nvidia.slack.com/archives/C0271E234TB/p1732609679973159?thread_ts=1732530842.185549&cid=C0271E234TB


if __name__ == '__main__':
    llm.export_ckpt(
        path='/tmp/checkpoints/llama3/HAHA/checkpoints/model_name=0--val_loss=0.00-step=19-consumed_samples=1280.0-last',
        target='hf',
        output_path=Path('/tmp/hf-ckpt-from-nemo/llama3-haha'),
        overwrite=True,
    )
