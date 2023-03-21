""" Script to process wikigraph data set into a standardized format """

import argparse
from collections import defaultdict
import json
import os

import tqdm

from wikigraphs.data.paired_dataset import ParsedDataset


def process_raw_data(dataset_path: str, split_name: str) -> None:
    """ Processes and saves raw wikigraph data into a standardized format """
    file_name = f"{split_name}.gz"
    split_data = defaultdict(list)
    print(f"Processing {split_name} data")
    for data_point in tqdm.tqdm(
        ParsedDataset(subset=split_name, data_dir=dataset_path, version='max1024')
    ):
        split_data['text'] = data_point.text
        split_data['nodes'] = data_point.graph._nodes
        edges, edge_index = [], []
        for edge in data_point.graph._edges:
            edges.append(edge[2])
            edge_index.append([edge[0], edge[1]])
        split_data['edges'].append(edges)
        split_data['edge_index'].append(edge_index)
    processed_file_name = f"{split_name}.json"
    processed_file_path = os.path.join(dataset_path, processed_file_name)
    with open(processed_file_path, 'w') as split_file:
        json.dump(split_data, split_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='processing raw wikigraph data ')
    parser.add_argument('--dataset-path', type=str, required=True)
    args = parser.parse_args()
    for split_name in ['train', 'valid', 'test']:
        process_raw_data(args.dataset_path, split_name)
