# Adapted from https://github.com/CryhanFang/CLIP2Video/blob/main/evaluation/metrics.py
import json
import numpy as np
import os
from typing import Dict, Any
from flagevalmm.registry import EVALUATORS


def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics["R1"] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics["R5"] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics["R10"] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics["MR"] = np.median(ind) + 1
    metrics["MedianR"] = metrics["MR"]
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics


def json_save(content: Dict[str, Any], jf_nm: str) -> None:
    with open(jf_nm, "w") as jf:
        json.dump(content, jf)


# Function to get evaluation metrics
def get_result(sim_matrix) -> Dict[str, Any]:
    """
    Get video retrieval evaluation metrics
    :param similarity_matrix: Similarity matrix
    :return: Dictionary of evaluation metrics
    """
    # compute text-to-video retrieval
    tv_metrics = compute_metrics(sim_matrix)
    # compute video-to-text retrieval
    vt_metrics = compute_metrics(sim_matrix.T)

    result = {
        "v2t_R@1": vt_metrics["R1"],
        "v2t_R@5": vt_metrics["R5"],
        "v2t_R@10": vt_metrics["R10"],
        "t2v_R@1": tv_metrics["R1"],
        "t2v_R@5": tv_metrics["R5"],
        "t2v_R@10": tv_metrics["R10"],
    }

    return result


@EVALUATORS.register_module()
class RetrievalMSRVTTEvaluator:
    def __init__(self, **kwargs):
        pass

    def process(self, dataset, output_dir, **kwargs) -> Dict[str, Any]:
        dataset_name = dataset.name

        # Load similarity matrix
        sim_matrix = np.load(os.path.join(output_dir, f"{dataset_name}.npy"))

        # Calculate retrieval metrics
        result = get_result(sim_matrix)

        # Save result
        json_save(result, os.path.join(output_dir, f"{dataset_name}_result.json"))
        print(f"{result}")

        return result
