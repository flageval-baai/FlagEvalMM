from ast import literal_eval
from typing import Dict, List


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: A list or string representing the first box [x1, y1, x2, y2]
        box2: A list or string representing the second box [x1, y1, x2, y2]

    Returns:
        float: IoU value between 0.0 and 1.0
    """
    if isinstance(box1, str):
        box1 = literal_eval(box1)
    if isinstance(box2, str):
        box2 = literal_eval(box2)

    # Check bounding box format
    if len(box1) != 4 or len(box2) != 4:
        raise ValueError("Bounding boxes must be in [x1, y1, x2, y2] format")

    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Compute intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # If there is no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate area
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union


def calculate_metrics(data):
    """
    Calculate evaluation metrics from prediction and ground truth pairs.

    Args:
        data: List of prediction dictionaries, each containing 'answer' and 'gt'.

    Returns:
        Dict: Evaluation metrics including average IoU, IoU at thresholds, and mean average precision.
    """
    iou_thresholds = [0.5, 0.75, 0.9]
    metrics = {
        "average_iou": 0.0,
        "iou_at_thresholds": {str(thresh): 0.0 for thresh in iou_thresholds},
        "average_precision": 0.0,
        "valid_items": 0,
    }

    for item in data:
        try:
            pred_box = item["answer"]  # May be a string or list
            gt_box = item["gt"]  # May be a string or list

            iou = calculate_iou(pred_box, gt_box)
            item["iou"] = iou
            metrics["average_iou"] += iou
            metrics["average_precision"] += iou  # Simplified mAP

            for thresh in iou_thresholds:
                metrics["iou_at_thresholds"][str(thresh)] += iou >= thresh

            metrics["valid_items"] += 1

        except (ValueError, KeyError, SyntaxError) as e:
            print(f"Skipping invalid item: {e}")

    # Compute averages
    if metrics["valid_items"] > 0:
        metrics["average_iou"] /= metrics["valid_items"]
        metrics["average_precision"] /= metrics["valid_items"]
        for thresh in iou_thresholds:
            metrics["iou_at_thresholds"][str(thresh)] /= metrics["valid_items"]

    return metrics


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """
    Evaluate model prediction results.

    Args:
        annotations: Ground truth annotation dictionary.
        predictions: List of model prediction dictionaries.

    Returns:
        Dict: Dictionary containing evaluation metrics.
    """
    result = []
    for idx, pred in enumerate(predictions):
        question_id = str(pred["question_id"])
        gt_data = annotations[question_id]
        gt_answer = gt_data["answer"]
        pred["gt"] = gt_answer
        pred["width"] = gt_data["width"]
        pred["height"] = gt_data["height"]
        result.append(pred)

    acc = calculate_metrics(result)
    return acc
