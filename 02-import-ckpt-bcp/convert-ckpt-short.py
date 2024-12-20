from pathlib import Path

## /opt/NeMo/nemo/collections/llm/api.py
from nemo.collections import llm

# Main guard needed, otherwise hang. Fix courtesy of @Chen Cui, @Ao Tang.
# NOTE: this best practice is now documented on Nemo-2.0 documentation.

MODEL = {
    'hf://google/gemma-2b': {
        'config_cls': llm.GemmaConfig2B,
        'model_cls': llm.GemmaModel,
        'output_path_prefix': '/tmp/checkpoints',   # On tmpfs, so we don't need to manually purge.
    },

    'hf://meta-llama/Meta-Llama-3-8B': {
        'config_cls': llm.Llama3Config8B,
        'model_cls': llm.LlamaModel,
        'output_path_prefix': '/tmp/checkpoints',   # On tmpfs, so we don't need to manually purge.
    },

    'hf://meta-llama/Meta-Llama-3-70B': {
        'config_cls': llm.Llama3Config70B,
        'model_cls': llm.LlamaModel,
        'output_path_prefix': '/result',           # Remember to manually purge to free 400+ GB.
    }
}


if __name__ == '__main__':
    ####
    #config = llm.GemmaConfig2B()
    #model = llm.GemmaModel(config=config)
    #source = "hf://google/gemma-2b"
    #
    # config = llm.Llama3Config8B()
    # model = llm.LlamaModel(config=config)
    # source = ""hf://meta-llama/Meta-Llama-3-8B"
    ####

    #source = 'hf://google/gemma-2b'
    source = 'hf://meta-llama/Meta-Llama-3-8B'
    #source = 'hf://meta-llama/Meta-Llama-3-70B'

    config_cls = MODEL[source]['config_cls']
    model_cls = MODEL[source]['model_cls']
    print(f'{config_cls=}')
    print(f'{model_cls=}')

    config = config_cls()
    model = model_cls(config=config)
    print(f'{config=}')
    print(f'{model=}')

    llm.import_ckpt(
        model=model,
        source=source,
        output_path=Path(f"{MODEL['output_path_prefix']}/{source[5:]}"),  # Drop 'hf://'
        overwrite=True,
    )
