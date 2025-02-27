from typing import Dict, List
from flagevalmm.evaluator.common_types import evaluate_multiple_choice
from flagevalmm.evaluator.pre_process import normalize_string
from word2number import w2n
from collections import defaultdict
import numpy as np
import re

MCA_QUESTION_TYPES = set(
    [
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_rel_distance",
        "route_planning",
        "obj_appearance_order",
    ]
)
NA_QUESTION_TYPES = set(
    [
        "object_abs_distance",
        "object_counting",
        "object_size_estimation",
        "room_size_estimation",
    ]
)


def fuzzy_matching(pred):
    # extract the first number or digits
    res = re.search(r"\d+\.\d+|\d+", pred)
    if res:
        return res.group()
    else:
        return pred.split(" ")[0].rstrip(".").strip()


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    if pred is None or target is None:
        return 0.0
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


def to_float(pred):
    try:
        pred = float(pred)
    except BaseException:
        pred = None
    return pred


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    results = defaultdict(lambda: {"num": 0, "correct": 0})

    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        question_type = gt["question_type"]
        if question_type in MCA_QUESTION_TYPES:
            is_correct = evaluate_multiple_choice(gt, pred)
        elif question_type in NA_QUESTION_TYPES:
            pred["raw_answer"] = pred["answer"]
            normalized_pred = fuzzy_matching(normalize_string(pred["answer"]))
            try:
                normalized_pred = w2n.word_to_num(normalized_pred)
            except BaseException:
                normalized_pred = normalized_pred
            pred["answer"] = normalized_pred
            is_correct = mean_relative_accuracy(
                to_float(normalized_pred),
                to_float(gt["answer"]),
                start=0.5,
                end=0.95,
                interval=0.05,
            )
        else:
            raise NotImplementedError
        pred["correct"] = is_correct
        pred["label"] = gt["answer"]
        pred["question_type"] = question_type
        results["avg"]["num"] += 1
        results["avg"]["correct"] += is_correct
        results[question_type]["num"] += 1
        results[question_type]["correct"] += is_correct
    for question_type, result in results.items():
        result["accuracy"] = round(result["correct"] / result["num"] * 100, 2)
    results["accuracy"] = results["avg"]["accuracy"]
    return results
