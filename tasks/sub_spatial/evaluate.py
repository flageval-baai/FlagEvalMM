from typing import Dict, List
from flagevalmm.evaluator.common_types import evaluate_multiple_choice
from flagevalmm.evaluator.pre_process import normalize_string
from word2number import w2n
import numpy as np
import re


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


def safe_acc(correct, num):
    return round(float(correct) / num * 100, 2) if num > 0 else 0.0


def cal_correct(gt, pred):
    question_type = gt["question_type"]
    if question_type == "multiple-choice":
        is_correct = evaluate_multiple_choice(gt, pred)
    elif question_type == "numerical":
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
    return round(is_correct, 2)


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    subtask = ["distance", "direction", "count", "depth_estimation", "size_estimation"]
    sources = ["EmbSpatial", "BLINK", "CV-Bench", "VSI-Bench"]

    results = {"avg": {"accuracy": 0.0, "correct": 0.0, "num": 0}}
    for source in sources:
        results[source] = {"avg": {"accuracy": 0.0, "correct": 0.0, "num": 0}}
        for task in subtask:
            results[source][task] = {"accuracy": 0.0, "correct": 0.0, "num": 0}

    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        source = gt["source"]
        task = gt["sub_task"]
        is_correct = cal_correct(gt, pred)

        pred["correct"] = is_correct
        pred["label"] = gt["answer"]
        pred["sub_task"] = task
        pred["source"] = source

        results["avg"]["num"] += 1
        results["avg"]["correct"] += is_correct

        results[source]["avg"]["num"] += 1
        results[source]["avg"]["correct"] += is_correct

        results[source][task]["num"] += 1
        results[source][task]["correct"] += is_correct

    results["avg"]["accuracy"] = safe_acc(
        results["avg"]["correct"], results["avg"]["num"]
    )

    for source in sources:
        results[source]["avg"]["accuracy"] = safe_acc(
            results[source]["avg"]["correct"], results[source]["avg"]["num"]
        )
        for task in subtask:
            results[source][task]["accuracy"] = safe_acc(
                results[source][task]["correct"], results[source][task]["num"]
            )

    for source in sources:
        for task in subtask:
            if results[source][task]["num"] == 0:
                del results[source][task]
    return results
