from typing import Dict, List
from collections import defaultdict
import json
import matplotlib.pyplot as plt

def cal_accuracy(
    annotations: Dict, predictions: List[Dict], target_source: str
) -> float:
    
    return 0

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Each box is a list of four float values: [x_min, y_min, x_max, y_max].
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate intersection coordinates
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    # Calculate intersection area
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate areas of each box
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    # Calculate union area
    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0

    return inter_area / union_area

def center_to_corners(box, image_path):

    image = plt.imread(image_path)
    img_height, img_weith = image.shape[:2]
    x1, y1, x2, y2 = box
    
    x_min = round(x1 / img_weith, 3)
    y_min = round(y1 / img_height, 3 )
    x_max = round(x2 / img_weith, 3)
    y_max = round(y2 / img_height, 3)
    return [x_min, y_min, x_max, y_max]

def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """评估模型预测结果
    
    Args:
        annotations: 标注数据
        predictions: 模型预测结果
        
    Returns:
        Dict: 包含评估指标的字典
    """
    results = defaultdict(lambda: {"num": 0, "correct": 0})

    ious = []
    acc_50 = 0
    acc_75 = 0
    acc_90 = 0    

    for pred in predictions:
        question_id = str(pred["question_id"])
        
        gt = annotations[question_id]
        answer_box = [float(x) for x in json.loads(gt["answer"])]
        image_path = pred["img_path"]
        
        
        # 歸一化
        pred_box = center_to_corners(json.loads(pred["answer"]), img_path)
        
        print(answer_box, pred_box)
        
        iou = calculate_iou(answer_box, pred_box)
        print(iou)
        pred["iou"] = iou
        ious.append(iou)

        if iou > 0.5:
            acc_50 += 1
        if iou > 0.75:
            acc_75 += 1
        if iou > 0.90:
            acc_90 += 1

    avg_iou = sum(ious) / len(ious) if ious else 0

    results = {
        "Num of Samples": len(predictions),
        "Average IoU": avg_iou,
        "Accuracy > 50%": acc_50 / len(predictions) * 100 if predictions else 0,
        "Accuracy > 75%": acc_75 / len(predictions) * 100 if predictions else 0,
        "Accuracy > 90%": acc_90 / len(predictions) * 100 if predictions else 0
    }
    
    
    return results