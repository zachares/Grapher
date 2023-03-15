import argparse
import ast
import json
import os
from typing import Any, Dict, List

import networkx as nx
import tqdm


class ASTWalker(ast.NodeVisitor):
    def __init__(self):
        self.node_id = 0
        self.stack = []
        self.graph = nx.Graph()
        self.nodes = {}

    def generic_visit(self, node):
        node_name = self.node_id
        self.node_id += 1
        # if available, extract AST node attributes
        try:
            docstring = ast.get_docstring(node, clean=False)
        except TypeError:
            docstring = None
        node_dict = {
            'docstring': docstring,
            'name': getattr(node, 'name', None),
            'args': getattr(node, 'arg', None),
            's': getattr(node, 's', None),
            'n': getattr(node, 'n', None),
            'id_': getattr(node, 'id', None),
            'attribute': getattr(node, 'attr', None),
            'type': type(node).__name__
        }
        self.nodes[node_name] = {
            key: value if any(value is dtype for dtype in [None, int, float]) else str(value)
            for key, value in node_dict.items()
        }
        self.nodes[node_name] = {
            key: value for key, value in self.nodes[node_name].items() if value is not None
        }
        # DFS traversal logic
        parent_name = None
        if self.stack:
            parent_name = self.stack[-1]
        self.stack.append(node_name)
        self.graph.add_node(node_name)
        if parent_name != None:
            # replicate AST as NetworkX object
            self.graph.add_edge(node_name, parent_name)
        super().generic_visit(node)
        self.stack.pop()


def get_split_data(dataset_path: str, split_name: str) -> List[Dict[str, Any]]:
    split_dataset = []
    print(f"Processing {split_name} set")
    for split_file_name in os.listdir(dataset_path):
        if split_name not in split_file_name or not split_file_name.endswith('.jsonl'):
            continue
        with open(os.path.join(dataset_path, split_file_name), encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                example = json.loads(line.strip())
                docstring = example['docstring']
                code_str = example['code']
                try:
                    tree = ast.parse(code_str)
                except:
                    continue
                walker = ASTWalker()
                walker.visit(tree)
                ast_nodes, ast_edges = walker.nodes, walker.graph.edges()
                ast_edges = [list(edge) for edge in ast_edges]
                node_ids = list(ast_nodes.keys())
                assert node_ids == list(range(len(node_ids)))
                ast_nodes = list(ast_nodes.values())
                split_dataset.append({
                    'docstring': docstring,
                    'code': code_str,
                    'nodes': ast_nodes,
                    'edges': ast_edges
                })
    return split_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--dataset-path", type=str, required=True)
    args = parser.parse_args()
    dataset = {}
    processed_dataset_path = os.path.join(args.dataset_path, "processed")
    os.makedirs(processed_dataset_path, exist_ok=True)
    for split_name in ['test', 'valid', 'train']:
        split_dataset = get_split_data(args.dataset_path, split_name)
        print(f"{split_name}ing set size {len(split_dataset)} graphs")
        with open(os.path.join(processed_dataset_path, f"{split_name}.json"), 'w') as split_file:
            json.dump(split_dataset, split_file)
