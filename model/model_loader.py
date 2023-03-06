import enum
import os

from transformers import T5ForConditionalGeneration

from data.graph_tokenizer import GraphTokenizer
from text2graph_generation_models import LitText2SerializedGraphLLM


class LanguageModelFactory(enum.Enum):
    """ A enum of text conditional generators from the hugging face language_model repository """
    t5 = T5ForConditionalGeneration


def init_model(
    tokenizer: GraphTokenizer,
    language_model_type: str,
    language_model_name: str,
    model_dir: str,
    learning_rate: float
) -> LitText2SerializedGraphLLM:
    return LitText2SerializedGraphLLM(
        language_model=LanguageModelFactory[language_model_type].value.from_pretrained(
            pretrained_model_name_or_path=language_model_name,
            cache_dir=model_dir
        ),
        tokenizer=tokenizer,
        model_dir=model_dir,
        learning_rate=learning_rate
    )


def load_model(model_dir: str) -> LitText2SerializedGraphLLM:
    checkpoint_model_path = os.path.join(model_dir, 'last.ckpt')
    return LitText2SerializedGraphLLM.load_from_checkpoint(checkpoint_model_path)
