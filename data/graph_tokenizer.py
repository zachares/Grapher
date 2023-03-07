import enum
from functools import partial
import itertools
from typing import List, Tuple

import numpy as np
import torch
from transformers import BatchEncoding, PreTrainedTokenizer, T5Tokenizer
from tqdm import tqdm


#pylint: disable=invalid-name
class TokenizerFactory(enum.Enum):
    """ A enum of text tokenizers from the hugging face transformers repository """
    t5 = T5Tokenizer


def _build_inputs_with_special_tokens(
    token_ids_0: List[int],
    _ : List[int],
    eos_token_id: int,
    pad_token_id: int,
) -> List[int]:
    return [pad_token_id] + token_ids_0 + [eos_token_id]


class GraphTokenizer():
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer
    ):
        self.tokenizer = tokenizer
        self.node_token = '__node__'
        self.edge_token = '__edge__'
        self.no_edge_token = '__no_edge__'
        self.tokenizer.add_tokens(self.node_token)
        self.tokenizer.add_tokens(self.edge_token)
        self.tokenizer.add_tokens(self.no_edge_token)
        self.add_special_tokens = partial(
            _build_inputs_with_special_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        self.tokenizer.build_inputs_with_special_tokens = self.add_special_tokens


    def encode_text(self, text: List[str]) -> List[List[int]]:
        """ Returns a set of text strings a list of list of token ids, one list for each string """
        return self.tokenizer(text, add_special_tokens=True, padding=False)['input_ids']

    def encode_graphs(self, serialized_graphs: List[List[str]]) -> List[List[List[int]]]:
        """ Returns a set of serialized graphs as a list of list of list of token ids
            the outer list is the list of graphs, the next list down is the list of edges
            in each graph and the inner most list is the list of token ids for each edge
        """
        return [
            self._encode_graph(serialized_graph)
            for serialized_graph in tqdm(serialized_graphs)
        ]

    def _encode_graph(self, serialized_graph: List[str]) -> List[List[int]]:
        """ Returns a list of list of token ids for a serialized graphs, the outer list is the list
            of edges in the graph, the inner list is the token ids of each edge. Each graph has
            an additional element in the outer list with the token ids required for generating
            a token id sequence for the entire graph
        """
        edges = self.tokenizer(
            serialized_graph,
            add_special_tokens=False,
            padding=False
        )['input_ids']
        return edges + [[self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]]

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
        graphs_token_ids: List[List[List[int]]],
        shuffle_edges: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.batch_token_ids([
            self._sequence_graph(graph_token_ids, shuffle_edges)
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

    def _sequence_graph(
        self,
        edges_token_ids: List[List[int]],
        shuffle_edges: bool
    ) -> List[int]:
        assert len(edges_token_ids[-1]) == 2
        shuffled_edges_token_ids = (
            np.random.permutation(edges_token_ids[:-1]).tolist()
            if shuffle_edges else edges_token_ids[:-1]
        )
        return (
            [edges_token_ids[-1][0]]
            + list(itertools.chain.from_iterable(shuffled_edges_token_ids))
            + [edges_token_ids[-1][1]]
        )
