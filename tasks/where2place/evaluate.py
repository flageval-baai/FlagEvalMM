from typing import Dict, List
from PIL import Image
from collections import defaultdict
import os.path as osp

from flagevalmm.evaluator.point_utils import text2pts, calculate_mask_score


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    results = defaultdict(lambda: {"num": 0, "score": 0})
    for pred in predictions:
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
        except Exception:
            continue
        mask_img = Image.open(osp.join(gt["data_root"], gt["mask_path"]))
        acc = calculate_mask_score(points, mask_img)
        # draw_point_result(gt, mask_img, acc, points)

        pred["score"] = acc
        pred["label"] = gt["mask_path"]

        results["avg"]["num"] += 1
        results["avg"]["score"] += acc
        question_type = gt.get("sub_task")
        results[question_type]["num"] += 1
        results[question_type]["score"] += acc
    for question_type, result in results.items():
        if result["num"]:
            result["accuracy"] = round(result["score"] / result["num"] * 100, 2)
        else:
            result["accuracy"] = 0.0
    results["accuracy"] = results["avg"]["accuracy"]
    return results
