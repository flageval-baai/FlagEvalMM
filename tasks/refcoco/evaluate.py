import re
from typing import Dict, List
from collections import defaultdict


COCO_REC_METRICS = [
    "IoU",
    "ACC@0.1",
    "ACC@0.3",
    "ACC@0.5",
    "ACC@0.7",
    "ACC@0.9",
    "Center_ACC",
]


def parse_float_sequence_within(input_str):
    """
    Extract the first sequence of four floating-point numbers within square brackets from a string.

    Args:
    input_str (str): A string that may contain a sequence of four floats within square brackets.

    Returns:
    list: A list of four floats if the pattern is found, or a list of four zeros if the pattern is not found.
    """
    pattern = r"[\(\[]?\s*(-?\d+(?:\.\d+)?)\s*,?\s*(-?\d+(?:\.\d+)?)\s*,?\s*(-?\d+(?:\.\d+)?)\s*,?\s*(-?\d+(?:\.\d+)?)\s*[\)\]]?"
    match = re.search(pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    return [0, 0, 0, 0]


def optimize_bbox(model, bbox, img_width, img_height):
    """
    Optimization for the Qwen model bbox output.
    """
    if model == "qwen2":
        if len(bbox) == 4 and (bbox[2] > img_width or bbox[3] > img_height):
            bbox = [num / 1000 for num in bbox]
    elif model == "qwen2.5":
        if any([x < 0 - 1e-5 or x > 1 + 1e-5 for x in bbox]):
            bbox = [
                x / img_width if i % 2 == 0 else x / img_height
                for i, x in enumerate(bbox)
            ]
    return bbox


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area


def compute_accuracy(box1, box2, threshold=0.5):
    """
    Compute the accuracy of two bounding boxes based on a specified threshold.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - threshold (float): Threshold for the IoU to consider the prediction correct.

    Returns:
    - float: Accuracy of the prediction based on the IoU threshold.
    """
    iou = compute_iou(box1, box2)
    return iou >= threshold


def compute_center_accuracy(box1, box2):
    """
    Compute if the center point of box 2 is within box 1.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - bool: True if the center point of box 2 is within box 1, False otherwise.
    """
    center_x = (box2[0] + box2[2]) / 2
    center_y = (box2[1] + box2[3]) / 2
    return box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3]


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    results = defaultdict(lambda: {"num": 0, "correct": 0})
    scorers = {
        "ACC@0.1": lambda x, y: compute_accuracy(x, y, 0.1),
        "ACC@0.3": lambda x, y: compute_accuracy(x, y, 0.3),
        "ACC@0.5": lambda x, y: compute_accuracy(x, y, 0.5),
        "ACC@0.7": lambda x, y: compute_accuracy(x, y, 0.7),
        "ACC@0.9": lambda x, y: compute_accuracy(x, y, 0.9),
        "Center_ACC": compute_center_accuracy,
    }

    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        pred["raw_answer"] = pred["answer"]
        pred["img_path"] = gt["img_path"]
        pred["answer"] = optimize_bbox(
            "",
            parse_float_sequence_within(pred["answer"]),
            gt["image_width"],
            gt["image_height"],
        )
        for scorer_name, scorer in scorers.items():
            score = scorer(gt["answer"], pred["answer"])
            results[scorer_name]["num"] += 1
            results[scorer_name]["correct"] += int(score)
        pred["label"] = gt["answer"]
        pred["IOU"] = compute_iou(gt["answer"], pred["answer"])
        pred["correct"] = pred["IOU"] >= 0.1
        results["avg"]["num"] += 1
        results["avg"]["correct"] += int(pred["IOU"] >= 0.1)
    for accuracy_type, result in results.items():
        result["accuracy"] = round(result["correct"] / result["num"] * 100, 2)
    results["accuracy"] = results["avg"]["accuracy"]
    return results
