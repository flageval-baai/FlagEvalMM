import re
from collections import defaultdict
from typing import Dict
from flagevalmm.registry import EVALUATORS
from flagevalmm.evaluator import BaseEvaluator

def extract_answer(answer)->str:
    """
    Extract the answer from the answer string.
    """
    # Extract answer from \boxed{} pattern if it exists
    boxed_pattern = r"\\boxed\{(.*?)\}"
    match = re.search(boxed_pattern, answer)
    if match:
        return match.group(1).strip()
    
    # Otherwise look for "answer is" pattern
    answer_patterns = [
        r"(?:answer|Answer|ANSWER)(?:\s+is:?\s*|\s*:?\s+)([-\d\w\s.]+)",
        r"(?:答案为:?\s*)([-\d\w\s.]+)"
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, answer)
        if match:
            # Clean up extracted answer
            extracted = match.group(1).strip()
            # Remove punctuation
            extracted = re.sub(r'[.,;:!?]$', '', extracted)
            return extracted
            
    # If no patterns match, return original answer
    return answer.strip()


@EVALUATORS.register_module()
class BoxEvaluator(BaseEvaluator):

    def cal_accuracy(self, annotation, answers):
        right = 0
        for answer in answers:
            question_id = str(answer["question_id"])
            gt = annotation[question_id]

            pred = extract_answer(answer["answer"])
            is_correct = gt["answer"] == pred
            answer["extract_answer"] = pred
            right += is_correct
            answer["correct"] = is_correct
            answer["label"] = gt["answer"]
            answer["question_type"] = gt["question_type"]
        result = {"accuracy": round(right / len(answers) * 100, 2)}
        return result
