import torch
from torch import nn
from transformers import T5ForConditionalGeneration


class Text2SerializedGraphLLM(nn.Module):
    def __init__(
        self,
        transformer_class: T5ForConditionalGeneration,
        transformer_name: str,
        cache_dir: str
    ):
        super().__init__()
        self.transformer = transformer_class.from_pretrained(transformer_name, cache_dir=cache_dir)

    def forward(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        target_edges: torch.Tensor,
        target_edges_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.transformer(
            input_ids=text,
            attention_mask=text_mask,
            decoder_input_ids=target_edges,
            decoder_attention_mask=target_edges_mask
        ).logits

    def sample(
        self,
        text_token_ids: torch.Tensor,
        text_token_mask: torch.Tensor
    ) -> torch.Tensor :
        return self.transformer.generate(
            input_ids=text_token_ids,
            max_length=150,
            attention_mask=text_token_mask
        ).sequences[:, 1:]
