""" Preprocessing functions for the WebNLG2020 data set """

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import networkx as nx
import unicodedata as ud

from corpusreader.benchmark_reader import Benchmark
from corpusreader.benchmark_reader import select_files


def _read_webnlg_dataset(dataset_path: str, split_name: str) -> Dict[str, Any]:
    """ Reads a split of the web nlg data set from memory and returns it as a dictionary """
    b = Benchmark()
    split_path = os.path.join(os.getcwd(), 'webnlg-dataset', 'release_v3.0', 'en', split_name)
    if split_name == 'test':
        files = [(split_path, 'semantic-parsing-test-data-with-refs-en.xml')]
    else:
        files = select_files(split_path)
    b.fill_benchmark(files)
    b.b2json(dataset_path, f'{split_name}.json')
    return json.load(
        open(os.path.join(dataset_path, f'{split_name}.json'), encoding='utf-8')
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


def process_raw_data(dataset_path: str, split_name: str) -> None:
    """ Preprocesses and saves data set as a set of sequenced text graph pairs """
    split_dataset = _read_webnlg_dataset(dataset_path=dataset_path, split_name=split_name)
    split_data = {}
    triples_list, split_data['text'] = _extract_text_triples_pairs(split_dataset)
    split_data['nodes'], split_data['edges'], split_data['edge_index'] = _parse_triples(triples_list)
    processed_file_name = f"{split_name}.json"
    processed_file_path = os.path.join(dataset_path, processed_file_name)
    with open(processed_file_path, 'w') as split_file:
        json.dump(split_data, split_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='processing raw webnlg data')
    parser.add_argument('--dataset-path', type=str, required=True)
    args = parser.parse_args()
    for split_name in ['train', 'dev', 'test']:
        process_raw_data(args.dataset_path, split_name)
