import re
from typing import Dict, Tuple
from flagevalmm.evaluator.pre_process import process_multiple_choice


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
