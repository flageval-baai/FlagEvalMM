from typing import Dict, List
from PIL import Image
import numpy as np
import re
from collections import defaultdict
from io import BytesIO
import base64


def base64_to_mask(mask_base64):
    mask_bytes = base64.b64decode(mask_base64)
    mask_img = Image.open(BytesIO(mask_bytes))
    mask = np.array(mask_img) / 255.0
    return mask


def text2pts(text, width=640, height=480):
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []
    for match in matches:
        vector = [float(num) if "." in num else int(num) for num in match.split(",")]
        print(vector)
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


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    results = defaultdict(lambda: {"num": 0, "correct": 0})
    accuracy_list = []
    
    for idx, pred in enumerate(predictions):
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        
        try:
            pred["raw_answer"] = pred["answer"]
            points = text2pts(
                pred.get("answer", ""),
                width=gt["image_width"],
                height=gt["image_height"],
            )
            pred["answer"] = str(points)
            points_array = np.array(points)
        except Exception:
            continue
            
        mask_base64 = gt["mask_base64"]
        mask = base64_to_mask(mask_base64)
        
        # 确保 mask 是 2D
        if mask.ndim > 2:
            mask = mask[..., 0]  # 或 np.mean(mask, axis=-1)
        
        acc = 0.0
        if len(points) > 0:
            in_range = (
                (points_array[:, 0] >= 0) & (points_array[:, 0] < mask.shape[1]) &
                (points_array[:, 1] >= 0) & (points_array[:, 1] < mask.shape[0])
            )
            valid_points = points_array[in_range]
            
            if len(valid_points) > 0:
                acc = mask[valid_points[:, 1], valid_points[:, 0]].mean()
        
        accuracy_list.append(acc)
        pred["correct"] = acc
        pred["label"] = gt["mask_path"]
        
        results["avg"]["num"] += 1
        results["avg"]["correct"] += acc
        
        question_type = gt.get("sub_task")
        results[question_type]["num"] += 1
        results[question_type]["correct"] += acc
    
    # 计算准确率
    for question_type, result in results.items():
        if result["num"] > 0:
            result["accuracy"] = round(result["correct"] / result["num"] * 100, 2)
        else:
            result["accuracy"] = 0.0
    
    results["accuracy"] = results["avg"]["accuracy"]
    return results
