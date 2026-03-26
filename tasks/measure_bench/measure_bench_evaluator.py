from typing import Dict, List, Union
from collections import defaultdict
from flagevalmm.evaluator import BaseEvaluator
from flagevalmm.registry import EVALUATORS
import re
import math
import unicodedata


def is_chinese(text):
    """Check if the text contains Chinese characters"""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


def normalize_string(text: str):
    # replace spacial characters
    replace_dict = {"′": "'", " ": " ", "‐": "-", "−": "-", "–": "-", "⋅": "·"}
    for k, v in replace_dict.items():
        text = text.replace(k, v)
    return text


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from a string: integers, decimals, and fractions (e.g., 3/4).
    Fractions are converted to floats. Returns numbers in the order they appear.
    """

    # Support normal minus '-' and Unicode minus '−'
    def norm_minus(s: str) -> str:
        return s.replace("\u2212", "-")  # U+2212 MINUS SIGN → hyphen-minus

    # Fractions first (to avoid capturing their parts separately), then decimals, then integers
    pattern = re.compile(
        r"""
        (?P<fraction>[+\-\u2212]?\d+\s*[/]\s*[+\-\u2212]?\d+)   # e.g., -3/4 or 3 / -5
        |
        (?P<decimal>[+\-\\u2212]?(?:\d*\.\d+|\d+\.\d*))         # e.g., .5, 0.5, 2., 3.14
        |
        (?P<integer>[+\-\\u2212]?\d+)                           # e.g., -7, 42
    """,
        re.VERBOSE,
    )

    out: List[float] = []
    for m in pattern.finditer(text):
        kind = m.lastgroup
        s = norm_minus(m.group(0)).strip()

        if kind == "fraction":
            # Split numerator/denominator with optional whitespace around '/'
            num_str, den_str = re.split(r"\s*/\s*", s, maxsplit=1)
            try:
                num = int(num_str)
                den = int(den_str)
                if den != 0:
                    out.append(num / den)
                # If denominator is zero, silently skip.
            except ValueError:
                # If something odd sneaks in, skip this token.
                continue
        else:
            # decimal or integer
            try:
                out.append(float(s))
            except ValueError:
                continue

    return out


def extract_answer(pred: Dict) -> str:
    indicators = ["Answer:", "Answer", "答案：", "答案:", "答案"]
    for indicator in indicators:
        if indicator in pred["answer"]:
            return pred["answer"].split(indicator)[-1].strip()
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    boxed_match = re.search(boxed_pattern, pred["answer"])
    if boxed_match:
        return boxed_match.group(1).strip()
    return pred["answer"]


def time_to_seconds(match):
    scale = [3600, 60, 1]
    seconds = 0
    for i in range(len(match)):
        if match[i] != "":
            seconds += scale[i] * int(match[i])
    return seconds


def _interval_matching(
    pred: Dict, interval: List[Union[float, str]], units: List[str]
) -> Dict:

    pred_str = pred["answer"]

    eval_result = {
        "all_correct": 0,
        "number_correct": 0,
        "number_error_rate": None,
        "unit_correct": 0,
    }
    pred_str_lower = pred_str.lower()
    # Special-case normalization: unify MICRO SIGN 'µ' (U+00B5) and GREEK MU 'μ' (U+03BC)
    pred_str_lower = unicodedata.normalize("NFKC", pred_str_lower).replace("µ", "μ")

    for unit in units:
        unit_lower = unicodedata.normalize("NFKC", unit.lower()).replace("µ", "μ")
        if unit_lower in pred_str_lower:
            eval_result["unit_correct"] = 1
    if len(units) == 0:
        eval_result["unit_correct"] = None
    # Time interval
    if isinstance(interval[0], str):
        left_interval = time_to_seconds(interval[0].split(":"))
        right_interval = time_to_seconds(interval[1].split(":"))
        time_pattern = r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b"

        # Find all time matches in the prediction string
        matches = re.findall(time_pattern, pred_str)

        if not matches:
            return eval_result
        match = matches[0]
        if len(match) < 2 or len(match) > 3:
            return eval_result
        pred_ans = time_to_seconds(match)
    else:
        # Number interval
        left_interval = interval[0]
        right_interval = interval[1]
        result = extract_numbers(pred_str)

        if len(result) == 0 or math.isinf(result[-1]) or math.isnan(result[-1]):
            return eval_result
        pred_ans = float(result[-1])
    if pred_ans < left_interval or pred_ans > right_interval:
        eps = 1e-6
        eval_result["number_error_rate"] = min(
            abs((pred_ans - left_interval) / (left_interval + eps)),
            abs((pred_ans - right_interval) / (right_interval + eps)),
        )
        return eval_result

    eval_result["number_correct"] = 1
    eval_result["number_error_rate"] = 0
    if eval_result["unit_correct"] is None or eval_result["unit_correct"] == 1:
        eval_result["all_correct"] = 1

    return eval_result


def interval_matching(
    pred: Dict, interval: List[Union[float, str]], units: List[str]
) -> int:
    return _interval_matching(pred, interval, units)


def is_current_better(result, best_result):
    if best_result is None:
        return True

    cur_unit_correct = (
        result["unit_correct"] if result["unit_correct"] is not None else 1
    )
    best_unit_correct = (
        best_result["unit_correct"] if best_result["unit_correct"] is not None else 1
    )
    if (
        result["number_correct"] + cur_unit_correct
        > best_result["number_correct"] + best_unit_correct
    ):
        return True
    if (
        result["number_error_rate"] is not None
        and result["number_error_rate"] < best_result["number_error_rate"]
    ):
        return True
    return False


def multi_interval_matching(
    pred: Dict, intervals: List[List[Union[float, str]]], units: List[str]
) -> Dict:
    best_result = None
    if len(units) == 0:
        units = [[]] * len(intervals)
    for interval, unit in zip(intervals, units):
        result = _interval_matching(pred, interval, unit)
        if result["all_correct"] == 1:
            return result
        if is_current_better(result, best_result):
            best_result = result
    return best_result


@EVALUATORS.register_module()
class MeasureBenchEvaluator(BaseEvaluator):
    def __init__(
        self,
        tracker_type,
        tracker_subtype=None,
        **kwargs,
    ):
        self.tracker_type = tracker_type
        self.tracker_subtype = tracker_subtype
        super().__init__(**kwargs)

    def get_score(self, gt: Dict, pred: Dict) -> Union[float, List[float]]:
        evaluator = gt["evaluator"]
        pred["raw_answer"] = pred["answer"]
        pred["answer"] = normalize_string(extract_answer(pred))
        registed_evaluator = set(["interval_matching", "multi_interval_matching"])
        if evaluator not in registed_evaluator:
            raise ValueError(f"Unsupported evaluator: {evaluator}")
        return eval(evaluator)(pred, **gt["evaluator_kwargs"])

    def cal_accuracy(
        self, annotations: Dict, predictions: List[Dict], *args, **kwargs
    ) -> Dict:
        class ScoreTracker:
            def __init__(self):
                self.total_score = 0
                self.total_number = 0
                self.total_number_with_unit = 0
                self.total_predicted_number = 0
                self.number_error_rate = 0
                self.number_correct = 0
                self.unit_correct = 0
                self.accuracy = 0
                self.number_accuracy = 0
                self.unit_accuracy = 0
                self.subtypes = defaultdict(
                    lambda: {
                        "total_score": 0,
                        "total_number": 0,
                        "total_number_with_unit": 0,
                        "total_predicted_number": 0,
                        "number_error_rate": 0,
                        "number_correct": 0,
                        "unit_correct": 0,
                        "number_accuracy": 0,
                        "unit_accuracy": 0,
                        "overall_accuracy": 0,
                    }
                )

            def update(self, eval_result, sub_type):
                sub_type = sub_type.lower()
                score = eval_result["all_correct"]
                self.total_score += eval_result["all_correct"]
                self.total_number += 1
                number_correct = eval_result["number_correct"]
                self.number_correct += number_correct
                if eval_result["number_error_rate"] is not None:
                    self.number_error_rate += eval_result["number_error_rate"]
                    self.total_predicted_number += 1
                    self.subtypes[sub_type]["total_predicted_number"] += 1
                    self.subtypes[sub_type]["number_error_rate"] += eval_result[
                        "number_error_rate"
                    ]
                unit_correct = eval_result["unit_correct"]
                if unit_correct is not None:
                    self.total_number_with_unit += 1
                    self.unit_correct += unit_correct
                    self.subtypes[sub_type]["unit_correct"] += unit_correct
                    self.subtypes[sub_type]["total_number_with_unit"] += 1

                self.subtypes[sub_type]["total_score"] += score
                self.subtypes[sub_type]["total_number"] += 1
                self.subtypes[sub_type]["number_correct"] += number_correct

            def update_accuracy(self):
                self.accuracy = round(self.total_score / self.total_number, 3)
                self.number_accuracy = round(self.number_correct / self.total_number, 3)
                if self.total_number_with_unit > 0:
                    self.unit_accuracy = round(
                        self.unit_correct / self.total_number_with_unit, 3
                    )
                else:
                    self.unit_accuracy = 1
                if self.total_predicted_number > 0:
                    self.number_error_rate = round(
                        self.number_error_rate / self.total_predicted_number, 3
                    )
                else:
                    self.number_error_rate = 0
                for sub_type in self.subtypes:
                    self.subtypes[sub_type]["overall_accuracy"] = round(
                        self.subtypes[sub_type]["total_score"]
                        / self.subtypes[sub_type]["total_number"],
                        3,
                    )
                    self.subtypes[sub_type]["number_accuracy"] = round(
                        self.subtypes[sub_type]["number_correct"]
                        / self.subtypes[sub_type]["total_number"],
                        3,
                    )
                    if self.subtypes[sub_type]["total_number_with_unit"] > 0:
                        self.subtypes[sub_type]["unit_accuracy"] = round(
                            self.subtypes[sub_type]["unit_correct"]
                            / self.subtypes[sub_type]["total_number_with_unit"],
                            3,
                        )
                    else:
                        self.subtypes[sub_type]["unit_accuracy"] = 1
                    if self.subtypes[sub_type]["total_predicted_number"] > 0:
                        self.subtypes[sub_type]["number_error_rate"] = round(
                            self.subtypes[sub_type]["number_error_rate"]
                            / self.subtypes[sub_type]["total_predicted_number"],
                            3,
                        )
                    else:
                        self.subtypes[sub_type]["number_error_rate"] = 0

            def to_serialize_dict(self):
                result = {
                    "total_score": self.total_score,
                    "total_number": self.total_number,
                    "number_correct": self.number_correct,
                    "unit_correct": self.unit_correct,
                    "total_number_with_unit": self.total_number_with_unit,
                    "accuracy": self.accuracy,
                    "number_accuracy": self.number_accuracy,
                    "unit_accuracy": self.unit_accuracy,
                    "number_error_rate": self.number_error_rate,
                }
                if self.subtypes:
                    result["subtypes"] = dict(self.subtypes)
                return result

        results = {}
        scores_by_type = defaultdict(ScoreTracker)

        for pred in predictions:
            question_id = str(pred["question_id"])
            gt = annotations[question_id]
            eval_result = self.get_score(gt, pred)
            # print(pred, gt, eval_result)
            pred.update(gt)

            pred["eval_result"] = eval_result

            # Update scores
            tracker = scores_by_type[pred[self.tracker_type]]
            tracker_overall = scores_by_type["overall"]
            if self.tracker_subtype is not None:
                tracker.update(
                    eval_result,
                    pred[self.tracker_subtype],
                )
                tracker_overall.update(
                    eval_result,
                    pred[self.tracker_subtype],
                )
            else:
                tracker.update(
                    eval_result,
                    pred[self.tracker_type],
                )
                tracker_overall.update(
                    eval_result,
                    pred[self.tracker_type],
                )
        # Calculate accuracy
        for tracker in scores_by_type.values():
            tracker.update_accuracy()
        # Convert ScoreTracker objects to the expected format
        for qtype, tracker in scores_by_type.items():
            results[qtype] = tracker.to_serialize_dict()
            # results[qtype].pop("subtypes")

        return results
