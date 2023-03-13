""" Data loading and processing classes for loading a data set of knowledge graph text pairs """

from copy import deepcopy
from typing import List, Tuple

import json
import random
from torch.utils.data import DataLoader, Dataset
import torch

from data.graph_tokenizer import GraphTokenizer
from data.preprocess_data import get_processed_data_paths


def init_dataloader(
    tokenizer: GraphTokenizer,
    split_name: str,
    shuffle_data: bool,
    data_path: str,
    batch_size: int,
    num_data_workers: int,
    augment_data: bool
) -> DataLoader:
    dataset = GraphDataset(
        tokenizer=tokenizer,
        processed_data_path=data_path,
        split_name=split_name,
        augment_data=augment_data
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset._collate_fn,
        num_workers=num_data_workers,
        shuffle=shuffle_data,
        pin_memory=True
    )


class GraphDataset(Dataset):
    def __init__(
        self,
        tokenizer: GraphTokenizer,
        processed_data_path: str,
        split_name: str,
        augment_data: bool
    ):
        self.tokenizer = tokenizer
        self.processed_data_path = processed_data_path
        self.augment_data = augment_data
        self.load_split_data(split_name)


    def load_split_data(self, split_name: str) -> None:
        self.split_name = split_name
        data_file_paths = get_processed_data_paths(
            dataset_path=self.processed_data_path,
            split_name=self.split_name,
            augment_data=self.augment_data
        )
        with open(data_file_paths['text_token_ids'], 'r', encoding='utf-8') as f:
            self.text_token_ids = json.load(f)

        with open(data_file_paths['graphs_token_ids'], 'r', encoding='utf-8') as f:
            self.graphs_token_ids = json.load(f)

    def __len__(self):
        return len(self.text_token_ids)

    def __getitem__(self, index: int) -> Tuple[List[List[int]], List[List[int]]]:
        return deepcopy(self.text_token_ids[index]), deepcopy(self.graphs_token_ids[index])

    def _collate_fn(
        self,
        data: Tuple[List[List[List[int]]], List[List[List[int]]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.augment_data:
            text_token_ids = []
            graph_token_ids = []
            for (text_list, edge_list) in data:
                text_token_ids.append(text_list.pop(random.randrange(len(text_list))))
                graph_token_ids.append(edge_list)
                if len(text_list) > 0:
                    text_token_ids.append(text_list.pop(random.randrange(len(text_list))))
                    random.shuffle(edge_list)
                    graph_token_ids.append(edge_list)
        else:
            text_token_ids = [point[0][0] for point in data]
            graph_token_ids = [point[1] for point in data]
        collated_data = self.tokenizer.batch_token_ids(text_token_ids)
        collated_data += self.tokenizer.batch_graphs_token_ids(graph_token_ids)
        return collated_data
