""" Data loading and processing classes for loading a data set of knowledge graph text pairs """

from typing import List, Tuple

import json
from torch.utils.data import DataLoader, Dataset
import torch

from graph_tokenizer import GraphTokenizer
from preprocess_data import get_processed_data_paths


def init_dataloader(
    tokenizer: GraphTokenizer,
    split_name: str,
    shuffle_data: bool,
    data_path: str,
    batch_size: int,
    num_data_workers: int
) -> DataLoader:
    dataset = GraphDataset(
        tokenizer=tokenizer,
        processed_data_path=data_path,
        split_name=split_name
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
        shuffle_edges: bool = False
    ):
        self.tokenizer = tokenizer
        self.processed_data_path = processed_data_path
        self.split_name = split_name
        self.shuffle_edges = shuffle_edges

        data_file_paths = get_processed_data_paths(
            dataset_path=self.processed_data_path,
            split_name=self.split_name
        )
        with open(data_file_paths['raw_text'], 'r', encoding='utf-8') as f:
            self.text = f.read().splitlines()

        with open(data_file_paths['text_token_ids'], 'r', encoding='utf-8') as f:
            self.text_token_ids = json.load(f)

        with open(data_file_paths['raw_graphs'], 'r', encoding='utf-8') as f:
            self.graphs = json.load(f)

        with open(data_file_paths['graphs_token_ids'], 'r', encoding='utf-8') as f:
            self.graphs_token_ids = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index: int) -> Tuple[List[List[int]], List[List[List[int]]]]:
        item = (self.text_token_ids[index], self.graphs_token_ids[index])
        return item

    def _collate_fn(
        self,
        data: Tuple[List[List[int]], List[List[List[int]]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_token_ids, graph_token_ids = data
        collated_data = self.tokenizer.batch_token_ids(text_token_ids)
        collated_data += self.tokenizer.batch_graphs_token_ids(graph_token_ids)
        return collated_data
