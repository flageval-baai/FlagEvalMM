from typing import Dict, List
from collections import defaultdict
import numpy as np
import re


def process_punctuation(inText):
    outText = inText
    punct = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]
    commaStrip = re.compile("(\d)(,)(\d)")  # noqa: W605
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")  # noqa: W605
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (
            re.search(commaStrip, inText) is not None
        ):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


def yes_or_no_extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if "yes" in words and "no" not in words:
        return "yes"
    if "yes" not in words and "no" in words:
        return "no"
    return "unknown"


def cal_accuracy(annotations, predictions):
    right = 0
    for answer in predictions:
        question_id = str(answer["question_id"])
        gt = annotations[question_id]
        answer["raw_answer"] = answer["answer"]
        answer["answer"] = yes_or_no_extraction(answer["answer"])
        is_correct = gt["answer"] == answer["answer"]
        answer["correct"] = is_correct
        answer["label"] = gt["answer"]
        answer["mix_figure_id"] = (
            f"{gt['subcategory']}_{gt['set_id']}_{gt['figure_id']}"
        )
        answer["mix_question_id"] = (
            f"{gt['subcategory']}_{gt['set_id']}_{gt['question_id_ori']}"
        )
        right += is_correct
    return round(right / len(predictions) * 100, 2)


def cal_mix_accuracy(predictions, mix_type):
    res = defaultdict(list)
    for answer in predictions:
        res[answer[mix_type]].append(answer["correct"])
    return round(np.mean([np.all(x) for x in res.values()]) * 100, 2)


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """
    Args:
        dataset (Dataset): dataset instance
        output_dir: str
    """
    results = {}
    results["question_acc"] = cal_accuracy(annotations, predictions)
    results["question_pair_acc"] = cal_mix_accuracy(predictions, "mix_question_id")
    results["figure_acc"] = cal_mix_accuracy(predictions, "mix_figure_id")
    results["accuracy"] = results["figure_acc"]
    return results
