import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

# Path to the directory containing your model and tokenizer files
MODEL_DIRECTORY = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        'pretrained_model'
    )
)


def load_model_and_tokenizer(model_directory: str = MODEL_DIRECTORY) -> (LlamaForCausalLM, LlamaTokenizerFast):
    """
    Load both the model (responsible for generating outputs) and the tokenizer (responsible for transforming inputs
    into readable vectors for the model)

    Args:
        model_directory (str): path reference pointing to the models folder

    Returns:
        model (LlamaForCausalLM): pre-trained model loaded from the models folder
        tokenizer (LlamaTokenizerFast): tokenizer loaded from the models folder

    """
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory, torch_dtype=torch.float16)
    return model, tokenizer
