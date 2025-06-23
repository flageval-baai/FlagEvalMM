import re
from typing import Dict, Tuple, Any
from flagevalmm.evaluator.pre_process import process_multiple_choice, normalize_string
import numpy as np
from word2number import w2n


def maybe_clean_answer(answer: str) -> str:
    if len(answer) == 1:
        return answer.upper()
    answer = process_multiple_choice(answer)
    return answer


def evaluate_multiple_choice(gt: Dict, pred: Dict) -> bool:
    pred["raw_answer"] = pred["answer"]
    pred["answer"] = maybe_clean_answer(pred["answer"])
    if len(pred["answer"]) > 1:
        pred["answer"] = pred["answer"][0]
    is_correct: bool = gt["answer"].upper() == pred["answer"]
    return is_correct


def evaluate_multiple_response(gt: Dict, pred: Dict) -> Tuple[bool, str]:
    cleaned_answer = maybe_clean_answer(pred["answer"])
    answer_list: list[str] = re.findall("[ABCDEFGHI]", cleaned_answer)

    cleaned_answer = "".join(sorted(set(answer_list)))
    pred["answer"] = cleaned_answer
    is_right: bool = gt["answer"].upper() == cleaned_answer
    return is_right, cleaned_answer


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
    if target == 0:
        return abs(pred - target)
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
    return bool(accuracy_score >= acceptance_threshold)


def is_yes_no_answer_correct(model_answer: str, label_answer: str) -> bool:
    gt_lower = label_answer.strip().lower()
    # Check if this is a binary yes/no question
    if gt_lower in ["yes", "no"]:
        return model_answer.lower().startswith(gt_lower)
    else:
        return False


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
