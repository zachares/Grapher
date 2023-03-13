import enum
import itertools
import random
from typing import List, Tuple

import torch
from transformers import BatchEncoding, PreTrainedTokenizer, T5Tokenizer
from tqdm import tqdm


#pylint: disable=invalid-name
class TokenizerFactory(enum.Enum):
    """ A enum of text tokenizers from the hugging face transformers repository """
    t5 = T5Tokenizer


class GraphTokenizer():
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.node_token = '__node__'
        self.edge_token = '__edge__'
        self.no_edge_token = '__no_edge__'
        self.tokenizer.add_tokens(self.node_token)
        self.tokenizer.add_tokens(self.edge_token)
        self.tokenizer.add_tokens(self.no_edge_token)

    def encode_grouped_text(
        self,
        texts_list: List[List[str]],
        add_special_tokens: bool
    ) -> List[List[List[int]]]:
        return [
            self._encode_text(text_list, add_special_tokens=add_special_tokens)
            for text_list in tqdm(texts_list)
        ]

    def _encode_text(self, text: List[str], add_special_tokens: bool) -> List[List[int]]:
        """ Returns a set of text strings a list of list of token ids, one list for each string """
        token_ids_list = self.tokenizer(text, padding=False)['input_ids']
        return [
            self._add_special_tokens(token_ids)
            if add_special_tokens else token_ids
            for token_ids in token_ids_list
        ]

    def decode_graphs(self, graph_token_ids: torch.Tensor) -> List[List[str]]:
        graphs_text_list = self.decode_text(graph_token_ids)
        triples_decoded_list = []
        for graph_text in graphs_text_list:
            triples_decoded = []
            for edge in graph_text.split(self.no_edge_token):
                if edge == '':
                    continue
                split_edge = [
                    text.split(self.edge_token) for text in edge.split(self.node_token)
                    if text != ''
                ]
                split_edge = list(itertools.chain.from_iterable(split_edge))
                split_edge = [text for text in split_edge if text != '']
                triples_decoded.append(split_edge)
            triples_decoded_list.append(triples_decoded)
        return triples_decoded_list

    def decode_text(self, text_token_ids: torch.Tensor) -> List[str]:
        text_decoded = []
        for token_ids in text_token_ids:
            bos_mask = (token_ids == self.tokenizer.pad_token_id).nonzero(as_tuple=False)
            eos_mask = (token_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=False)
            if (
                (bos_mask[0] + 1) < eos_mask[0]
                and eos_mask[0] < token_ids.size(0)
                and (bos_mask[0] + 1) < token_ids.size(0)
            ):
                text_dec = self.tokenizer._decode(token_ids[bos_mask[0] + 1:eos_mask[0]])
            else:
                text_dec = ""
            text_decoded.append(text_dec)
        return text_decoded

    def batch_token_ids(
        self,
        text_token_ids: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_id_batch = BatchEncoding({'input_ids': text_token_ids})
        padded_ids_dict = self.tokenizer.pad(
            token_id_batch,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return padded_ids_dict['input_ids'], padded_ids_dict['attention_mask']

    def batch_graphs_token_ids(
        self,
        graphs_token_ids: List[List[List[int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.batch_token_ids([
            self._add_special_tokens(list(itertools.chain.from_iterable(graph_token_ids)))
            for graph_token_ids in graphs_token_ids
        ])

    def serialize_graphs(
        self,
        nodes_list: List[List[str]],
        edges_list: List[List[str]],
        edge_indexes_list: List[List[List[int]]]
    ) -> List[List[str]]:
        """ Serializes a graph / represents a graph as a sequences of edges, each edge is
            expressed by a string. Returns a list of the serialized graphs
        """
        serialized_graphs = []
        for nodes, edges, edge_inds in zip(nodes_list, edges_list, edge_indexes_list):
            graph_text = []
            for edge, edge_ind in zip(edges, edge_inds):
                graph_text.append("".join([
                    self.node_token,
                    nodes[edge_ind[0]],
                    self.edge_token,
                    edge,
                    self.node_token,
                    nodes[edge_ind[1]],
                    self.no_edge_token
                ]))
            serialized_graphs.append(graph_text)
        return serialized_graphs

    def _add_special_tokens(self, token_ids: List[int]) -> List[int]:
        return [self.tokenizer.pad_token_id] + token_ids + [self.tokenizer.eos_token_id]
