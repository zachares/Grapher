import torch
import networkx as nx
import itertools
from WebNLG_Text_to_triples import Evaluation_script_json
import os
from misc.rdf import save_webnlg_rdf
import json
from typing import Callable, Dict, List
from transformers import T5Tokenizer

failed_node = 'failed node'
failed_edge = 'failed edge'
nonode_str = '__no_node__'


def compute_loss(
    criterion: Dict[str, Callable],
    logits_graphs: torch.Tensor,
    serialized_graphs_token_ids: torch.Tensor,
    padding_mask: torch.Tensor
) -> torch.Tensor:
    # --------- Node Loss ---------
    # predicts the next token in the sequence
    return criterion['ce'](
        logits_graphs[:, :-1].transpose(1,2),
        torch.where(padding_mask == 0, -100, serialized_graphs_token_ids)[:, 1:]
    ).mean()


def decode(cand, bos_token_id, eos_token_id, tokenizer, failed=failed_node):

    bos_mask = (cand == bos_token_id).nonzero(as_tuple=False)
    if len(bos_mask) > 0:
        eos_mask = (cand == eos_token_id).nonzero(as_tuple=False)
        if len(eos_mask) > 0:
            s = tokenizer._decode(cand[bos_mask[0] + 1:eos_mask[0]])
        else:
            s = failed
    else:
        s = failed

    return s


def decode_text(
    tokenizer: T5Tokenizer,
    text_input_ids: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int
) -> List[str]:

    text_decoded = []

    for text in text_input_ids:
        bos_mask = (text == bos_token_id).nonzero(as_tuple=False)
        eos_mask = (text == eos_token_id).nonzero(as_tuple=False)
        try:
            text_dec = tokenizer._decode(text[bos_mask[0] + 1:eos_mask[0]])
        except:
            text_dec = ""
        text_decoded.append(text_dec)
    return text_decoded


def decode_graph(
    edges_text_list: List[str],
    node_token: str,
    edge_token: str,
    no_edge_token: str
) -> List[List[str]]:
    triples_decoded_list = []
    for edges_text in edges_text_list:
        triples_decoded = []
        for edge in edges_text.split(no_edge_token):
            if edge == '':
                continue
            split_edge = [text.split(edge_token) for text in edge.split(node_token) if text != '']
            split_edge = list(itertools.chain.from_iterable(split_edge))
            split_edge = [text for text in split_edge if text != '']
            triples_decoded.append(split_edge)
        triples_decoded_list.append(triples_decoded)
    return triples_decoded_list


def compute_scores(hyp, ref, iteration, eval_dir, split, rank):
    refs = [[' | '.join(i) for i in t] for t in ref]
    hyps = [[' | '.join(i) for i in t] for t in hyp]
    categories = [' '] * len(refs)

    ref_fname, hyp_fname = save_webnlg_rdf(hyps, refs, categories, os.path.join(eval_dir, split), f'{iteration}_{rank}')

    scores_fname = os.path.join(eval_dir, split, f'scores_{iteration}_{rank}.json')

    Evaluation_script_json.main(ref_fname, hyp_fname, scores_fname)

    scores = json.load(open(scores_fname))
    scores = {'Precision': scores['Total_scores']['Exact']['Precision'],
              'Recall': scores['Total_scores']['Exact']['Recall'],
              'F1': scores['Total_scores']['Exact']['F1']}

    return scores
