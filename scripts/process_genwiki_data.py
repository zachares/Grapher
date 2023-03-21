""" Script to process genwiki data set into a standardized format """

import argparse
from collections import defaultdict
import json
import os
from typing import List

import tqdm

from process_tekgen_data import get_graph_features

def get_train_filepaths(dataset_path: str) -> List[str]:
    """ Returns the file paths for all the training data in the genwiki data set"""
    full_dataset_path = os.path.join(dataset_path, "train", "full")
    assert os.path.isdir(full_dataset_path)
    return [
        os.path.join(full_dataset_path, file_name)
        for file_name in os.listdir(full_dataset_path)
        if file_name.endswith('.json')
    ]


def add_entity_names(text: str, entities: List[str]) -> str:
    """ Returns a processed a passage from the genwiki data set by reinserting names in the text """
    split_text = []
    for sub_text in text.split('<'):
        split_text.extend(sub_text.split('>'))
    processed_text = []
    for sub_text in split_text:
        if len(sub_text) == 0:
            continue
        if sub_text.startswith('ENT'):
            index = int(sub_text.split('_')[-1])
            processed_text.append(entities[index])
        else:
            processed_text.append(sub_text)
    return "".join(processed_text)


def process_raw_data(dataset_path: str, split_name: str) -> None:
    """ Processes and saves raw genwiki data into a standardized format """
    if split_name == 'test':
        file_paths = [os.path.join(dataset_path, 'test', 'test.json')]
    elif split_name == 'train':
        file_paths = get_train_filepaths(dataset_path)
    split_data = defaultdict(list)
    print(f"Processing {split_name} data")
    for file_path in tqdm.tqdm(file_paths):
        with open(file_path) as file:
            data_shard = json.load(file)
            for data_point in data_shard:
                split_data['text'].append(
                    add_entity_names(data_point['text'], entities=data_point['entities'])
                )
                nodes, edges, edge_index = get_graph_features(data_point['graph'])
                split_data['nodes'].append(nodes)
                split_data['edges'].append(edges)
                split_data['edge_index'].append(edge_index)
    processed_file_name = f"{split_name}.json"
    processed_file_path = os.path.join(dataset_path, processed_file_name)
    with open(processed_file_path, 'w') as split_file:
        json.dump(split_data, split_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='processing raw genwiki data ')
    parser.add_argument('--dataset-path', type=str, required=True)
    args = parser.parse_args()
    for split_name in ['train', 'test']:
        process_raw_data(args.dataset_path, split_name)
