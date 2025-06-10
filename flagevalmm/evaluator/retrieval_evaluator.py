import json
import numpy as np
import os
from typing import Dict, Any
from flagevalmm.registry import EVALUATORS

# TODO: refactor code


def i2t(probs: np.ndarray, return_ranks: bool = False):
    npts = probs.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # captions per imae
    k = probs.shape[1] // probs.shape[0]

    for index in range(npts):
        inds = np.argsort(probs[index])[::-1]

        # Score
        rank = 1e20
        for i in range(k * index, k * index + k, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = float(100.0 * len(np.where(ranks < 1)[0]) / len(ranks))
    r5 = float(100.0 * len(np.where(ranks < 5)[0]) / len(ranks))
    r10 = float(100.0 * len(np.where(ranks < 10)[0]) / len(ranks))
    medr = float(np.floor(np.median(ranks)) + 1)
    meanr = float(ranks.mean() + 1)

    metrics = (r1, r5, r10, medr, meanr)
    if return_ranks:
        return metrics, (ranks, top1)
    return metrics


def t2i(probs: np.ndarray, return_ranks: bool = False):
    npts = probs.shape[0]
    # captions per imae
    k = probs.shape[1] // probs.shape[0]
    ranks = np.zeros(k * npts)
    top1 = np.zeros(k * npts)
    probs = probs.T

    for index in range(npts):
        for i in range(k):
            inds = np.argsort(probs[k * index + i])[::-1]
            ranks[k * index + i] = np.where(inds == index)[0][0]
            top1[k * index + i] = inds[0]

    # Compute metrics
    r1 = float(100.0 * len(np.where(ranks < 1)[0]) / len(ranks))
    r5 = float(100.0 * len(np.where(ranks < 5)[0]) / len(ranks))
    r10 = float(100.0 * len(np.where(ranks < 10)[0]) / len(ranks))
    medr = float(np.floor(np.median(ranks)) + 1)
    meanr = float(ranks.mean() + 1)

    metrics = (r1, r5, r10, medr, meanr)
    if return_ranks:
        return metrics, (ranks, top1)
    return metrics


def json_save(content: Dict[str, Any], jf_nm: str) -> None:
    with open(jf_nm, "w") as jf:
        json.dump(content, jf)


@EVALUATORS.register_module()
class RetrievalEvaluator:
    def __init__(self, **kwargs):
        pass

    def process(self, dataset, output_dir, **kwargs):
        dataset_name = dataset.name

        # Load similarity matrix
        sim_matrix = np.load(os.path.join(output_dir, f"{dataset_name}.npy"))

        # Dataset-specific shape validation
        if dataset_name == "f30k" and sim_matrix.shape != (1000, 5000):
            print(
                f"f30k_sim.shape: {sim_matrix.shape}, please check it. If in try-run mode, ignore the message"
            )

        # Calculate retrieval metrics
        result_i2t = i2t(sim_matrix)
        result_t2i = t2i(sim_matrix)

        # Print raw results
        print(f"{result_i2t}_{result_t2i}")

        # Prepare results dictionary
        content = {
            "i2t_R@1": result_i2t[0],
            "i2t_R@5": result_i2t[1],
            "i2t_R@10": result_i2t[2],
            "t2i_R@1": result_t2i[0],
            "t2i_R@5": result_t2i[1],
            "t2i_R@10": result_t2i[2],
            "mean_recall": (sum(result_i2t[:3]) + sum(result_t2i[:3])) / 6.0,
        }

        # Save results
        json_save(content, os.path.join(output_dir, f"{dataset_name}_result.json"))
        print(f"{content}")
