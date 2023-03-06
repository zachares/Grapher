""" Model classes for training a model to generate serialized graphs from text """

import itertools
import json
import os
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import PreTrainedModel

from WebNLG_Text_to_triples import Evaluation_script_json
from data.graph_tokenizer import GraphTokenizer
from data.rdf import save_webnlg_rdf


def compute_scores(
    graphs_generated: List[List[str]],
    graphs_ground_truth: List[List[str]],
    iteration: int,
    model_dir: str,
    split_name: str,
    rank: int
) -> Dict[str, float]:
    """ Computes evaluation metrics for comparing the actual and estimated knowledge graphs for a
        batch of text graph pairs
    """
    refs = [[' | '.join(i) for i in t] for t in graphs_ground_truth]
    hyps = [[' | '.join(i) for i in t] for t in graphs_generated]
    categories = [' '] * len(refs)

    ref_fname, hyp_fname = save_webnlg_rdf(
        hyps=hyps,
        refs=refs,
        categories=categories,
        out_dir=os.path.join(model_dir, split_name),
        iteration=f'{iteration}_{rank}'
    )

    scores_fname = os.path.join(model_dir, split_name, f'scores_{iteration}_{rank}.json')

    Evaluation_script_json.main(ref_fname, hyp_fname, scores_fname)

    scores = json.load(open(scores_fname))
    return {
        'Precision': scores['Total_scores']['Exact']['Precision'],
        'Recall': scores['Total_scores']['Exact']['Recall'],
        'F1': scores['Total_scores']['Exact']['F1']
    }


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
        self.save_hyperparameters()
        self.unlabelled_idx = -100
        self.model =  Text2SerializedGraphLLM(language_model)
        self.tokenizer = tokenizer
        self.model_dir=model_dir
        self.learning_rate = learning_rate

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """ Performs a forward pass through the model and computes the loss for a batch of
            text graph pairs in the trainings set
        """
        # graph_token_ids: batch_size X seq_len_node
        text_token_ids, text_token_attn_mask, graph_token_ids, graph_token_attn_mask = batch
        # graph_token_id_logits: batch_size X seq_len_node X vocab_size
        graph_token_id_logits = self.model(
            text_token_id=text_token_ids,
            text_token_attn_mask=text_token_attn_mask,
            graph_token_ids=graph_token_ids,
            graph_token_attn_mask=graph_token_attn_mask
        )
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
        graph_token_ids_generated = self.model.sample(text_token_ids, text_token_attn_mask)
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
        graphs_ground_truth = [tr[:10] for tr in graphs_generated]

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

        self.log_dict(scores)

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, List[List[str]]]:
        return self.eval_step(batch)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, List[List[str]]]:
        return self.eval_step(batch)

    def validation_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, 'valid')

    def test_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.learning_rate)
        return optimizer


class Text2SerializedGraphLLM(nn.Module):
    """ Text to SerializedGraph Generation Models """
    def __init__(self, language_model: PreTrainedModel):
        super().__init__()
        self.language_model = language_model
        self.max_length = 500 # legacy magic number from Grapher

    def forward(
        self,
        text_token_ids: torch.Tensor,
        text_token_mask: torch.Tensor,
        graph_token_ids: torch.Tensor,
        graph_token_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.language_model(
            input_ids=text_token_ids,
            attention_mask=text_token_mask,
            decoder_input_ids=graph_token_ids,
            decoder_attention_mask=graph_token_mask
        ).logits

    def sample(
        self,
        text_token_ids: torch.Tensor,
        text_token_mask: torch.Tensor
    ) -> torch.Tensor :
        return self.language_model.generate(
            input_ids=text_token_ids,
            max_length=self.max_length,
            attention_mask=text_token_mask
        ).sequences[:, 1:]
