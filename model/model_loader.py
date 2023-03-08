import enum
import os

from transformers import T5ForConditionalGeneration

from data.graph_tokenizer import GraphTokenizer
from model.text2graph_generation_models import LitText2SerializedGraphLLM


class LanguageModelFactory(enum.Enum):
    """ A enum of text conditional generators from the hugging face language_model repository """
    t5 = T5ForConditionalGeneration


def init_and_load_model(
    tokenizer: GraphTokenizer,
    language_model_type: str,
    language_model_name: str,
    model_dir: str,
    model_name: str,
    learning_rate: float,
    load_model: bool
) -> LitText2SerializedGraphLLM:
    model_hyperparameters = {
        "language_model": LanguageModelFactory[language_model_type].value.from_pretrained(
            pretrained_model_name_or_path=language_model_name,
            cache_dir=model_dir
        ),
        "tokenizer": tokenizer,
        "model_dir": model_dir,
        "learning_rate": learning_rate
    }
    if load_model:
        latest_checkpoint = ""
        for file_name in os.listdir(model_dir):
            if model_name in file_name:
                latest_checkpoint = (
                    latest_checkpoint
                    if latest_checkpoint > file_name and len(latest_checkpoint) >= len(file_name)
                    else file_name
                )
        checkpoint_model_path = os.path.join(model_dir, "vanilla_language_model-v2.ckpt")
        print(f"Loading model from {checkpoint_model_path}")
        model = LitText2SerializedGraphLLM.load_from_checkpoint(
            checkpoint_model_path,
            **model_hyperparameters
        )
    else:
        model = LitText2SerializedGraphLLM(**model_hyperparameters)
    return model
