from collections import defaultdict
import json
import os
import random
from typing import Dict, List

import nltk
nltk.download('punkt')

from data.rdf import save_webnlg_rdf
from WebNLG_Text_to_triples import Evaluation_script_json


def compute_scores(
    graphs_generated: List[List[str]],
    graphs_ground_truth: List[List[str]],
    iteration: int,
    model_dir: str,
    split_name: str,
    rank: int
) -> Dict[str, float]:
    """ Computes evaluation metrics for comparing the actual and estimated knowledge graphs for a
        batch of text graph pairs
    """
    refs = [[' | '.join(i) for i in t] for t in graphs_ground_truth]
    random.shuffle(graphs_ground_truth)
    hyps = [[' | '.join(i) for i in t] for t in graphs_ground_truth]
    hyps = [[' | '.join(i) for i in t] for t in graphs_generated]
    categories = [' '] * len(refs)

    ref_fname, hyp_fname = save_webnlg_rdf(
        hyps=hyps,
        refs=refs,
        categories=categories,
        out_dir=os.path.join(model_dir, split_name),
        iteration=f'{iteration}_{rank}'
    )

    scores_fname = os.path.join(model_dir, split_name, f'scores_{iteration}_{rank}.json')

    Evaluation_script_json.main(ref_fname, hyp_fname, scores_fname)

    scores = json.load(open(scores_fname))
    return {
        'Precision': float(scores['Total_scores']['Exact']['Precision']),
        'Recall': float(scores['Total_scores']['Exact']['Recall']),
        'F1': float(scores['Total_scores']['Exact']['F1'])
    }


def compute_graph_score(
    graphs_generated: List[List[str]],
    graphs_ground_truth: List[List[str]]
) -> Dict[str, float]:
    """ Computes evaluation metrics for comparing the actual and estimated knowledge graphs for a
        batch of text graph pairs
    """
    graph_overlap = []
    for graph_gen, graph_gt in zip(graphs_generated, graphs_ground_truth):
        total_edges_truth = len(graph_gen)
        total_edges_generated = len(graph_gt)
        correct_count = 0
        edges = set()
        edge_dict_gt = defaultdict(list)
        for idx, edge in enumerate(graph_gt):
            edge_dict_gt[tuple(edge)].append(idx)
            edges.add(tuple(edge))
        edge_dict_gen = defaultdict(list)
        for idx, edge in enumerate(graph_gen):
            edge_dict_gen[tuple(edge)].append(idx)
            edges.add(tuple(edge))
        for edge in list(edges):
            correct_count += min(len(edge_dict_gen[edge]), len(edge_dict_gt[edge]))
        assert correct_count <= total_edges_truth
        assert correct_count <= total_edges_generated
        graph_overlap.append(correct_count / max(total_edges_generated, total_edges_truth))
    return {'Graph Overlap': sum(graph_overlap) / len(graph_overlap)}
