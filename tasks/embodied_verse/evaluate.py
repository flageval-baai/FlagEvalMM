from typing import Dict, List, Any
from flagevalmm.evaluator.common_types import evaluate_multiple_choice
from flagevalmm.evaluator.pre_process import normalize_string
from word2number import w2n
import numpy as np
import re
from PIL import Image
import os.path as osp
from collections import defaultdict


def fuzzy_matching(pred: str):
    # extract the first number or digits
    res = re.search(r"\d+\.\d+|\d+", pred)
    if res:
        return res.group()
    else:
        return pred.split(" ")[0].rstrip(".").strip()


def to_float(pred):
    try:
        pred = float(pred)
    except (ValueError, TypeError):
        pred = None
    return pred


def abs_dist_norm(pred, target):
    return abs(pred - target) / abs(target)


def mean_relative_accuracy(pred, target, start, end, interval):
    if pred is None or target is None:
        return 0.0
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= (1 - conf_intervs)
    return accuracy.mean()


def normalize_answer(model_answer: str):
    normalized_answer = fuzzy_matching(normalize_string(model_answer))
    try:
        normalized_answer = w2n.word_to_num(normalized_answer)
    except BaseException:
        pass
    return normalized_answer


def is_numerical_answer_correct(
    model_answer: str,
    label_answer: float,
    confidence_start: float = 0.5,
    confidence_end: float = 0.95,
    confidence_interval: float = 0.05,
    acceptance_threshold: float = 0.5,
) -> bool:
    normalized_answer_float = to_float(model_answer)
    label_answer_float = to_float(label_answer)
    accuracy_score = mean_relative_accuracy(
        pred=normalized_answer_float,
        target=label_answer_float,
        start=confidence_start,
        end=confidence_end,
        interval=confidence_interval,
    )
    return bool(float(accuracy_score) > acceptance_threshold)


def text2pts(text, width=640, height=480):
    # Answer is in the last line
    text = text.strip().split("\n")[-1]
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []
    for match in matches:
        vector = [float(num) if "." in num else int(num) for num in match.split(",")]
        if len(vector) == 2:
            x, y = vector
            if isinstance(x, float) or isinstance(y, float):
                x = int(x * width)
                y = int(y * height)
            points.append((x, y))
        elif len(vector) == 4:
            x0, y0, x1, y1 = vector
            if isinstance(x0, float):
                x0 = int(x0 * width)
                y0 = int(y0 * height)
                x1 = int(x1 * width)
                y1 = int(y1 * height)
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = 1
            y, x = np.where(mask)
            points.extend(list(np.stack([x, y], axis=1)))
    return points


def is_point_answer_correct(
    model_answer_points: list, mask_img: Any, acceptance_threshold: float = 0.5
) -> bool:
    mask = np.array(mask_img) / 255.0
    points_array = np.array(model_answer_points)
    accuracy_score: float = 0.0
    if len(model_answer_points) > 0:
        in_range = (
            (points_array[:, 0] >= 0)
            & (points_array[:, 0] < mask.shape[1])
            & (points_array[:, 1] >= 0)
            & (points_array[:, 1] < mask.shape[0])
        )
        # Calculate the accuracy score based on the mask
        accuracy_score = float(
            np.concatenate(
                [
                    mask[points_array[in_range, 1], points_array[in_range, 0]],
                    np.zeros(points_array.shape[0] - in_range.sum()),
                ]
            ).mean()
        )
    return accuracy_score >= acceptance_threshold


def evaluate_point(gt: Dict, pred: Dict) -> bool:
    pred["raw_answer"] = pred["answer"]
    points = text2pts(
        pred.get("answer", ""),
        width=gt["image_width"],
        height=gt["image_height"],
    )
    pred["answer"] = str(points)
    mask_img = Image.open(osp.join(gt["data_root"], gt["mask_path"]))
    is_correct = is_point_answer_correct(points, mask_img)
    return is_correct


def evaluate_numerical(gt: Dict, pred: Dict) -> bool:
    pred["raw_answer"] = pred["answer"]
    pred["answer"] = normalize_answer(pred["answer"])
    is_correct = is_numerical_answer_correct(
        model_answer=pred["answer"], label_answer=gt["answer"]
    )
    return is_correct


def evaluate_yes_no(gt: Dict, pred: Dict) -> bool:
    pred["raw_answer"] = pred["answer"]
    pred["answer"] = normalize_string(pred["answer"].strip().split("\n")[-1])
    gt_lower = gt["answer"].strip().lower()
    # Check if this is a binary yes/no question
    if gt_lower in ["yes", "no"]:
        return pred["answer"].lower().startswith(gt_lower)
    else:
        return False


def cal_correct(gt, pred):
    if gt["question_type"] == "multiple-choice":
        is_correct = evaluate_multiple_choice(gt, pred)
    elif gt["question_type"] == "numerical":
        is_correct = evaluate_numerical(gt, pred)
    elif gt["question_type"] == "yes-no":
        is_correct = evaluate_yes_no(gt, pred)
    elif gt["question_type"] == "point":
        is_correct = evaluate_point(gt, pred)
    else:
        raise ValueError("Unknown question type")
    return is_correct


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    right: float = 0.0
    detailed_keys = ["level-1", "level-2"]
    detailed_results = defaultdict(list)

    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        is_correct = cal_correct(gt, pred)
        if isinstance(is_correct, bool):
            is_correct_as_float = float(is_correct)
        else:
            is_correct_as_float = is_correct
        pred["correct"] = is_correct_as_float
        pred["label"] = gt["answer"]
        right += is_correct_as_float
        if detailed_keys:
            for key in detailed_keys:
                detailed_results[gt[key]].append(is_correct_as_float)
    results = {
        "accuracy": round(right / len(predictions) * 100, 2),
    }
    if detailed_keys:
        for key, values in detailed_results.items():
            results[key] = round(sum(values) / len(values) * 100, 2)
    return results
