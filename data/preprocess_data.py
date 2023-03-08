""" Preprocessing functions for the WebNLG2020 data set """

from collections import defaultdict
import itertools
import json
import os
from typing import Any, Dict, List, Tuple

import networkx as nx
import unicodedata as ud

from corpusreader.benchmark_reader import Benchmark
from corpusreader.benchmark_reader import select_files
from data.graph_tokenizer import GraphTokenizer


def prepareWebNLG(
    tokenizer: GraphTokenizer,
    source_path: str,
    target_path: str,
    split_names: List[str],
    augment_data: bool
) -> None:
    """ Preprocesses and saves data set as a set of sequenced text graph pairs """
    for split_name in split_names:
        augment_bool = augment_data if split_name == "train" else False
        data_file_paths = get_processed_data_paths(
            dataset_path=target_path,
            split_name=split_name,
            augment_data=augment_bool
        )
        preprocessing_done = True
        for file_path in data_file_paths.values():
            if not os.path.isfile(file_path):
                preprocessing_done = False
                break
        if preprocessing_done:
            continue

        split_dataset = _read_webnlg_dataset(
            source_path=source_path,
            split_name=split_name,
            target_path=target_path
        )
        triples_list, text_list = _extract_text_triples_pairs(split_dataset)
        nodes_list, edges_list, edge_indexes_list = _parse_triples(triples_list)
        serialized_graphs_list = tokenizer.serialize_graphs(
            nodes_list=nodes_list,
            edges_list=edges_list,
            edge_indexes_list=edge_indexes_list
        )
        if augment_bool and split_name == 'train':
            text_list, serialized_graphs_list = _aggregate_duplicates(text_list, serialized_graphs_list)
        else:
            text_list = [[text] for text in text_list]

        text_token_ids = tokenizer.encode_grouped_text(text_list, add_special_tokens=True)
        graph_token_ids = tokenizer.encode_grouped_text(
            serialized_graphs_list,
            add_special_tokens=False
        )

        with open(data_file_paths['raw_text'], 'w', encoding='utf-8') as f:
            json.dump(text_list, f)

        with open(data_file_paths['raw_graphs'], 'w', encoding='utf-8') as f:
            json.dump(
                {'nodes': nodes_list, 'edges': edges_list, 'edge_indexes': edge_indexes_list},
                f
            )
        with open(data_file_paths['graphs_token_ids'], 'w', encoding='utf-8') as f:
            json.dump(graph_token_ids, f)

        with open(data_file_paths['text_token_ids'], 'w', encoding='utf-8') as f:
            json.dump(text_token_ids, f)


def _aggregate_duplicates(
    text_list: List[str],
    serialized_graphs_list: List[List[str]]
) -> Tuple[List[List[str]], List[List[str]]]:
    """ Aggregates text which is paired with the same knowledge graph """
    sequenced_graphs_list = ["".join(sorted(edges_str)) for edges_str in serialized_graphs_list]
    graph2idx = defaultdict(list)
    for idx, graph_str in enumerate(sequenced_graphs_list):
        graph2idx[graph_str].append(idx)
    text_idxs, graph_idxs = [], []
    for idxs in graph2idx.values():
        text_idxs.append(idxs)
        graph_idxs.append(idxs[0])
    return (
        [[text_list[idx] for idx in point_idxs] for point_idxs in text_idxs],
        [serialized_graphs_list[idx] for idx in graph_idxs]
    )


def get_processed_data_paths(
    dataset_path: str,
    split_name: str,
    augment_data: bool
) -> Dict[str, str]:
    """ Returns a dictionary of the file paths to each of the four files required for text to
        serialized graph generation
    """
    augmented_str = "augmented" if augment_data else "not_augmented"
    return {
        'raw_graphs': os.path.join(dataset_path, f"{split_name}_{augmented_str}_graphs.json"),
        'raw_text': os.path.join(dataset_path, f"{split_name}_{augmented_str}_text.json"),
        'graphs_token_ids': os.path.join(
            dataset_path,
            f"{split_name}_{augmented_str}_graphs_token_ids.json"
        ),
        'text_token_ids': os.path.join(
            dataset_path,
            f"{split_name}_{augmented_str}_text_token_ids.json"
        )
    }


def _read_webnlg_dataset(
    source_path: str,
    split_name: str,
    target_path: str
) -> Dict[str, Any]:
    """ Reads a split of the web nlg data set from memory and returns it as a dictionary """
    b = Benchmark()
    if split_name == 'test':
        files = [(
            os.path.join(source_path, split_name),
            'semantic-parsing-test-data-with-refs-en.xml'
        )]
    else:
        files = select_files(os.path.join(source_path, split_name))
    b.fill_benchmark(files)
    b.b2json(target_path, f'{split_name}.json')
    return json.load(
        open(os.path.join(target_path, f'{split_name}.json'), encoding='utf-8')
    )


def _extract_text_triples_pairs(
    dataset: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    """ Extracts the knowledge graph text pairs from a data set stored in a dictionary """
    normalize = lambda text: ud.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    triples_list = []
    text_list = []
    for ind, entry in enumerate(dataset['entries']):
        triples = entry[str(ind + 1)]['modifiedtripleset']
        proc_triples = []
        for triple in triples:
            obj, rel, sub = triple['object'], triple['property'], triple['subject']
            obj = normalize(obj.strip('\"').replace('_', ' '))
            sub = normalize(sub.strip('\"').replace('_', ' '))
            proc_triples.append(f'__subject__ {sub} __predicate__ {rel} __object__ {obj}')
        merged_triples = ' '.join(proc_triples)
        proc_lexs = [normalize(l['lex']) for l in entry[str(ind + 1)]['lexicalisations']]
        for lex in proc_lexs:
            text_list.append(
                normalize('summarize as a knowledge graph: ')
                + lex
            )
            triples_list.append(merged_triples)
    return triples_list, text_list


def _parse_triples(
    graph_data: List[str]
) -> Tuple[List[List[str]], List[List[str]], List[List[List[int]]]]:
    """ Processes and returns  a list of knowledge graphs represented as sequences of semantic / rdf
        triples as a list of nodes, edges and edge indexes.
    """
    all_nodes = []
    all_edges = []
    all_edges_ind = []
    for triples_str in graph_data:
        graph_nx = nx.DiGraph()
        triples_str += ' '
        for triple_str in triples_str.split('__subject__')[1:]:
            head = triple_str.split('__predicate__')[0][1:-1]
            relop = triple_str.split('__predicate__')[1].split('__object__')[0][1:-1]
            tail = triple_str.split('__predicate__')[1].split('__object__')[1][1:-1]
            graph_nx.add_edge(head, tail, edge=relop)
            graph_nx.nodes[head]['node'] = head
            graph_nx.nodes[tail]['node'] = tail
        nodes = list(graph_nx.nodes)
        edges = []
        edges_ind = []
        for u, v, d in graph_nx.edges(data=True):
            edges.append(d['edge'])
            edges_ind.append([nodes.index(u), nodes.index(v)])
        all_nodes.append(nodes)
        all_edges.append(edges)
        all_edges_ind.append(edges_ind)
    return all_nodes, all_edges, all_edges_ind
