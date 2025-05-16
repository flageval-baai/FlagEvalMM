from typing import Dict, List
from collections import defaultdict
import numpy as np
import re

def clean_text(text: str) -> str:
    """Remove punctuation, convert to lowercase, and normalize '&' to 'and'."""
    # Replace '&' with 'and' before removing other punctuation
    text = text.replace('&', ' and ')
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Normalize whitespace
    text = ' '.join(text.split())
    text = text.replace(" ", "").strip()
    return text

def process_dict_answer(answer: Dict, gt: Dict, data_type: str, category: str,
                       counters: Dict) -> None:
    """Process dictionary-type answers."""
    pred_answer = clean_text(answer["answer"])
    
    # Handle main_title which could be a string or a list
    main_title = gt["human_answer"] if gt["human_answer"] != "" else gt["answer"]['main_title']
    
    # Check if main_title is a list or a string
    if isinstance(main_title, list):
        # For list, check if all items in the list are in pred_answer
        is_correct = all(clean_text(item) in pred_answer for item in main_title)
    else:
        # For string, check if main_title is in pred_answer
        is_correct = clean_text(main_title) in pred_answer
    
    answer["is_correct"] = int(is_correct)
    answer["main_title"] = main_title
    
    # Count main title
    counters["type_correct"][data_type] += is_correct
    counters["type_total"][data_type] += 1
    
    if data_type in ["book", "game"] and category is not None:
        counters["category_correct"][data_type][category] += is_correct
        counters["category_total"][data_type][category] += 1

def calculate_accuracies(counters: Dict) -> Dict:
    """Calculate accuracy metrics from counters."""
    accuracies = {}
    
    # Data type accuracies
    for data_type in counters["type_total"]:
        accuracies[data_type] = {
            "accuracy": counters["type_correct"][data_type] / counters["type_total"][data_type],
            "correct": counters["type_correct"][data_type],
            "total": counters["type_total"][data_type]
        }

    # Category accuracies - only for book and game
    for data_type in ["book", "game"]:
        if data_type in counters["category_total"]:
            accuracies[f"{data_type}_by_category"] = {
                category: {
                    "accuracy": counters["category_correct"][data_type][category] / counters["category_total"][data_type][category],
                    "correct": counters["category_correct"][data_type][category],
                    "total": counters["category_total"][data_type][category]
                }
                for category in counters["category_total"][data_type]
            }

    # Overall accuracy
    total_correct = sum(counters["type_correct"].values())
    total_samples = sum(counters["type_total"].values())
    accuracies["overall"] = {
        "accuracy": total_correct / total_samples,
        "correct": total_correct,
        "total": total_samples
    }

    return accuracies

def cal_accuracy(annotations: Dict, predictions: List[Dict], edit_distance: bool = False) -> Dict:
    """Calculate accuracy metrics for predictions against ground truth annotations."""
    counters = {
        "type_correct": defaultdict(int),
        "type_total": defaultdict(int),
        "category_correct": defaultdict(lambda: defaultdict(int)),
        "category_total": defaultdict(lambda: defaultdict(int))
    }

    for answer in predictions:
        question_id = answer["question_id"]
        gt = annotations[question_id]
        data_type = gt["question_type"]
        category = gt.get("category", None) if data_type in ["book", "game"] else None

        process_dict_answer(answer, gt, data_type, category, counters)

        answer["data_type"] = data_type
        answer["category"] = category
        answer["gt_answer"] = gt["answer"]
        answer["img_path"] = gt["img_path"]
        answer["url"] = gt["url"]

    return calculate_accuracies(counters)

def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """Calculate evaluation results for predictions."""
    return cal_accuracy(annotations, predictions)
