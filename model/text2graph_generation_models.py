""" Model classes for training a model to generate serialized graphs from text """

import itertools
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from transformers import PreTrainedModel

from data.graph_tokenizer import GraphTokenizer
from metrics import compute_scores


class LitText2SerializedGraphLLM(pl.LightningModule):
    """ Pytorch Lightning Wrapper for Text2SerializedGraph Generation Models"""
    def __init__(
        self,
        language_model: PreTrainedModel,
        tokenizer: GraphTokenizer,
        model_dir: str,
        learning_rate: float
    ):
        super().__init__()
        self.unlabelled_idx = -100
        self.max_length = 200 # maximum sequence length in NLG data set is 191
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.model_dir=model_dir
        self.learning_rate = learning_rate

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int # required for pytorch lightning interface
    ) -> torch.Tensor:
        """ Performs a forward pass through the model and computes the loss for a batch of
            text graph pairs in the trainings set
        """
        # graph_token_ids: batch_size X seq_len_node
        text_token_ids, text_token_attn_mask, graph_token_ids, graph_token_attn_mask = batch
        # graph_token_id_logits: batch_size X seq_len_node X vocab_size
        graph_token_id_logits = self.language_model(
            input_ids=text_token_ids,
            attention_mask=text_token_attn_mask,
            decoder_input_ids=graph_token_ids,
            decoder_attention_mask=graph_token_attn_mask
        ).logits
        # predicting the next token in the sequence
        loss = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.unlabelled_idx)(
            input=graph_token_id_logits[:, :-1].transpose(1,2),
            target=torch.where(
                graph_token_attn_mask == 0,
                self.unlabelled_idx,
                graph_token_ids
            )[:, 1:]
        ).mean()
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=text_token_ids.size(0)
        )
        return loss

    def eval_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, List[List[str]]]:
        """ Returns a dictionary with a generated graph and the actual graph for every text
            graph pair in the batch
        """
        text_token_ids, text_token_attn_mask, graph_token_ids, _ = batch
        graph_token_ids_generated = self.language_model.generate(
            input_ids=text_token_ids,
            max_length=self.max_length,
            attention_mask=text_token_attn_mask
        )
        return {
            'graphs_ground_truth': self.tokenizer.decode_graphs(graph_token_ids),
            'graphs_generated': self.tokenizer.decode_graphs(graph_token_ids_generated)
        }

    def eval_epoch_end(
        self,
        outputs: List[Dict[str, List[List[str]]]],
        split_name: str
    ) -> None:
        """ Evaluates model performance at generating graphs from text and logs metrics"""
        iteration = self.global_step
        rank = self.global_rank
        graphs_ground_truth = list(
            itertools.chain.from_iterable([out['graphs_ground_truth'] for out in outputs])
        )
        graphs_generated = list(
            itertools.chain.from_iterable([out['graphs_generated'] for out in outputs])
        )
        # make sure number of paths is smaller than 10 (legacy hack from Grapher Project)
        graphs_ground_truth = [tr[:10] for tr in graphs_ground_truth]
        graphs_generated = [tr[:10] for tr in graphs_generated]
        scores = compute_scores(
            graphs_generated=graphs_generated,
            graphs_ground_truth=graphs_ground_truth,
            iteration=iteration,
            model_dir=self.model_dir,
            split_name=split_name,
            rank=rank
        )
        for k, v in scores.items():
            self.logger.experiment.add_scalar(f'{split_name}_score/{k}', v, global_step=iteration)
        self.log_dict(scores, prog_bar=True)

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int # required for pytorch lightning interface
    ) -> Dict[str, List[List[str]]]:
        return self.eval_step(batch)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int # required for pytorch lightning interface
    ) -> Dict[str, List[List[str]]]:
        return self.eval_step(batch)

    def validation_epoch_end(self, outputs: List[Dict[str, List[List[str]]]]):
        self.eval_epoch_end(outputs, 'valid')

    def test_epoch_end(self, outputs: List[Dict[str, List[List[str]]]]):
        self.eval_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.learning_rate)
