from typing import Dict, List
from flagevalmm.evaluator.common_types import evaluate_multiple_choice


def cal_accuracy(
    annotations: Dict, predictions: List[Dict], target_source: str
) -> float:
    right = 0
    total = 0
    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        if gt["task"] != target_source:
            continue
        total += 1
        is_correct = evaluate_multiple_choice(gt, pred)
        pred["correct"] = is_correct
        pred["label"] = gt["answer"]
        right += is_correct
    return round(right / (total + 1e-10) * 100, 2)


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    results = {}
    targets = {
        "Color Counting": "accuracy_color_counting",
        "Color Proportion": "accuracy_color_proportion",
        "Color Recognition": "accuracy_color_recognition",
        "Color Comparison": "accuracy_color_comparison",
        "Color Mimicry": "accuracy_color_mimicry",
        "Color Extraction": "accuracy_color_extraction",
        "Color Blindness": "accuracy_color_blindness",
        "Color Illusion": "accuracy_color_illusion",
        "Object Recognition": "accuracy_object_recognition",
        "Object Counting": "accuracy_object_counting",
    }
    for target, metric in targets.items():
        results[metric] = cal_accuracy(annotations, predictions, target)

    results["accuracy_perception"] = round(
        (results["accuracy_color_recognition"] + results["accuracy_color_extraction"]+ 
        results["accuracy_object_recognition"]) / 3, 2
    )
    results["accuracy_reasoning"] = round(
        (results["accuracy_color_counting"] + results["accuracy_color_proportion"]+
        results["accuracy_color_comparison"] + results["accuracy_color_mimicry"]+
        results["accuracy_color_blindness"] + results["accuracy_color_illusion"]+ 
        results["accuracy_object_counting"]) / 7, 2
    )
    
    results["accuracy"] = round(
        (results["accuracy_perception"] + results["accuracy_reasoning"]) / 2, 2
    )
    return results
