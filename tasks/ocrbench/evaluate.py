from typing import Dict, List
import unicodedata
from collections import defaultdict


def get_score(gt, pred):
    dataset_name = gt["dataset"]
    gt_ans = gt["answer"]

    # Convert single answer to list for unified processing
    answers = gt_ans if isinstance(gt_ans, list) else [gt_ans]

    def process(x, enable_normalize=False):
        if enable_normalize:
            x = unicodedata.normalize("NFKC", x)
        if dataset_name == "HME100k" or dataset_name == "HMER":
            return x.strip().replace("\n", " ").replace(" ", "")
        else:
            return x.lower().strip().replace("\n", " ")

    # Process prediction once
    processed_pred = process(pred)

    # Check if any answer matches the prediction
    return int(any(process(answer) in processed_pred for answer in answers))


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    results = {}
    ocrbench_score = defaultdict(lambda: [0, 0])
    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        score = get_score(gt, pred["answer"])
        pred["score"] = score
        pred["label"] = gt["answer"]
        pred["question_type"] = gt["question_type"]
        ocrbench_score[pred["question_type"]][0] += pred["score"]
        ocrbench_score[pred["question_type"]][1] += 1

    final_score = sum(ocrbench_score[cat][0] for cat in ocrbench_score.keys())
    results["final_score"] = [final_score, len(predictions)]
    results["accuracy"] = round(final_score / len(predictions) * 100, 3)
    results.update(ocrbench_score)
    return results
