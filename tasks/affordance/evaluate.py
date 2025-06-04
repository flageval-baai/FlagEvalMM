import json
from typing import Dict, List, Union
from ast import literal_eval


def calculate_iou(box1: Union[str, List[float]], box2: Union[str, List[float]]) -> float:
    """
    Compute the IoU (Intersection over Union) between two bounding boxes.

    The bounding boxes must be in [x1, y1, x2, y2] format.
    Input can be either a list of floats or a stringified list.
    """
    if isinstance(box1, str):
        box1 = literal_eval(box1)
    if isinstance(box2, str):
        box2 = literal_eval(box2)

    if len(box1) != 4 or len(box2) != 4:
        raise ValueError("Bounding boxes must be in [x1, y1, x2, y2] format")

    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union


def calculate_metrics(data: List[Dict], iou_thresholds: List[float] = [0.5, 0.75, 0.9]) -> Dict:
    """
    Calculate evaluation metrics including average IoU and simplified mAP.
    """
    metrics = {
        'average_iou': 0.0,
        'iou_at_thresholds': {str(thresh): 0.0 for thresh in iou_thresholds},
        'average_precision': 0.0,
        'valid_items': 0
    }

    for item in data:
        try:
            pred_box = item["answer"]
            gt_box = item["gt"]

            iou = calculate_iou(pred_box, gt_box)
            item['iou'] = iou
            metrics['average_iou'] += iou
            metrics['average_precision'] += iou  # simplified mAP

            for thresh in iou_thresholds:
                if iou >= thresh:
                    metrics['iou_at_thresholds'][str(thresh)] += 1

            metrics['valid_items'] += 1

        except (ValueError, KeyError, SyntaxError, TypeError) as e:
            print(f"[WARN] Skipped invalid item: {e}")

    if metrics['valid_items'] > 0:
        total = metrics['valid_items']
        metrics['average_iou'] /= total
        metrics['average_precision'] /= total
        for thresh in iou_thresholds:
            metrics['iou_at_thresholds'][str(thresh)] /= total

    return metrics


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """
    Evaluate predictions against ground truth.

    Args:
        annotations: Ground truth annotations keyed by question_id
        predictions: List of model prediction dicts
        output_path: Optional path to save detailed results

    Returns:
        Dict containing evaluation metrics
    """
    result = []
    for pred in predictions:
        question_id = str(pred["question_id"])
        gt_data = annotations.get(question_id, {})
        gt_answer = gt_data.get("answer")
        
        pred["gt"] = gt_answer
        pred["width"] = gt_data.get("width", None)
        pred["height"] = gt_data.get("height", None)
        result.append(pred)
    metrics = calculate_metrics(result)
    return metrics

