import nemo_run as run

from nemo.collections import llm
from nemo.collections.llm.recipes import llama3_8b
from nemo.collections.llm.gpt.data.mock import MockDataModule

def custom_llama3_8b():
    recipe = llama3_8b.finetune_recipe(name='HEHE', num_nodes=1, dir='/tmp/ftexp', peft_scheme=None)
    recipe.data = run.Config(MockDataModule, tokenizer=None, seq_length=8192, global_batch_size=512, micro_batch_size=1)
    print(recipe)
    # recipe.resume.restore_config.xxx
    #restore_config=<Config[RestoreConfig(path='nemo://meta-llama/Meta-Llama-3-8B')]>)]>,
    return recipe

if __name__ == "__main__":
    # When running this file, it will run the `custom_llama3_8b` recipe

    # To select the `custom_llama3_70b` recipe, use the following command:
    #   python custom_recipe.py --factory custom_llama3_70b
    #   This will automatically call the custom_llama3_70b that's defined above

    # Note that any parameter can be overwritten by using the following syntax:
    # python custom_recipe.py trainer.max_steps=2000

    # You can even apply transformations when triggering the CLI as if it's python code
    # python custom_recipe.py "trainer.max_steps*=2"

    run.cli.main(llm.finetune, default_factory=custom_llama3_8b)
