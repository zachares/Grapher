""" Script to process tekgen data set into a standardized format """

import argparse
from collections import defaultdict
import json
import os
from typing import List, Tuple

import tqdm

def get_graph_features(triples: List[List[str]]) -> Tuple[List[str], List[str], List[List[int]]]:
    """ Returns a list of the nodes, edges and the edge index of a knowledge graph described by
        a set of triples
    """
    nodes, edges, edge_index = set(), [], []
    for triple in triples:
        nodes.add(triple[0])
        nodes.add(triple[2])
        edges.append(triple[1])
    nodes = list(nodes)
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    for triple in triples:
        edge_index.append([node2idx[triple[0]], node2idx[triple[2]]])
    return nodes, edges, edge_index


def process_raw_data(dataset_path: str, split_name: str) -> None:
    """ Processes and saves raw tekgen data into a standardized format """
    file_name = f"quadruples-{split_name}.tsv"
    file_path = os.path.join(dataset_path, file_name)
    split_data = defaultdict(list)
    print(f"Processing {split_name} data")
    with open(file_path) as file:
        for line in tqdm.tqdm(file):
            data_point = json.loads(line)
            split_data['text'].append(data_point['sentence'])
            split_data['original_serialization'].append(data_point['serialized_triples'])
            nodes, edges, edge_index = get_graph_features(data_point['triples'])
            split_data['nodes'].append(nodes)
            split_data['edges'].append(edges)
            split_data['edge_index'].append(edge_index)
    processed_file_name = f"{split_name}.json"
    processed_file_path = os.path.join(dataset_path, processed_file_name)
    with open(processed_file_path, 'w') as split_file:
        json.dump(split_data, split_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='processing raw tekgen data ')
    parser.add_argument('--dataset-path', type=str, required=True)
    args = parser.parse_args()
    for split_name in ['train', 'val', 'test']:
        process_raw_data(args.dataset_path, split_name)
