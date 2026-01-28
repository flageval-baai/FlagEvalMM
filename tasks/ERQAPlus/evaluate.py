from typing import Dict, List, Union
from collections import defaultdict
import re


def strip_answer(answer):
    answer = re.sub("The", "", answer)
    answer = re.sub("If", "", answer)
    answer = re.sub("[INST]", "", answer)
    answer = re.sub("[/INST]", "", answer)
    answer = re.sub("<Img>", "", answer)
    answer = re.sub("</Img>", "", answer)
    answer = answer.strip()
    return answer


def remove_special_characters(text):
    pattern = r"[-`\\【】\*\$、,，。.；;:：？\?！!\s\n\u4e00-\u9fff0-9①②③④⑤⑥⑦\[\]\<>a-z=\'\"\(\)\{\}]+"
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text


def process_multiple_choice(answer):
    answer = strip_answer(answer)
    pattern = r"^([A-Z])\."
    matches = re.match(pattern, answer)
    if matches:
        return matches.group(1)
    key_words = [
        "boxed",
        "Answer:",
        "Answer is",
        "answer is",
        "option is",
        "Correct option",
        "correct option",
        "Answer",
        "answer",
        "故选",
        "选择",
        "正确选项为",
        "答案选",
        "答案为",
        "答案是",
        "因此",
        "答案",
        "答案：",
    ]

    for key_word in key_words:
        if key_word in answer:
            answer = answer.split(key_word)[-1]
            break
    answer = remove_special_characters(answer)
    # keep the last line
    answer = answer.split("\n")[-1]
    pattern = r"[A-Z]"
    matches = re.findall(pattern, answer)
    return ",".join(matches)


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


def key_items_matching(
    pred: Dict, key_items: List[List[str]], remove_space=False
) -> int:
    def process(x):
        if remove_space:
            x = x.replace(" ", "")
        return x.lower()

    # Check if any answer variant matches the prediction as a substring
    def match_any(pred_str, answer_variants):
        assert isinstance(answer_variants, list)
        return int(any(process(answer) in pred_str for answer in answer_variants))

    processed_pred = process(pred["answer"])
    if isinstance(key_items[0], list):
        return int(
            all(match_any(processed_pred, item_variants) for item_variants in key_items)
        )
    elif isinstance(key_items[0], str):
        return int(match_any(processed_pred, key_items))
    else:
        raise ValueError(f"Unsupported key_items type: {type(key_items[0])}")


def choices_matching(pred: Dict, label: str) -> int:
    pred["answer"] = process_multiple_choice(pred["answer"])
    pred_ans = pred["answer"]
    label = label.upper().replace(" ", "")
    if len(label) > 1:
        label = "".join(sorted(set(label)))
        pred_ans = "".join(sorted(set(pred["answer"])))
    # elif len(pred["answer"]) > 1:
    #     pred["answer"] = pred["answer"][0]
    #     pred_ans = pred["answer"]
    return int(label == pred_ans)
    # return 0


def ordered_list_matching(pred: dict, order) -> int:  # Strict order list mating

    pred_ans = (
        pred["answer"]
        .lower()
        .replace(" ", "")
        .replace("*", "")
        .replace("'", "")
        .strip(" `,")
    )

    if isinstance(order[0], list):
        return int(
            any(
                [
                    pred_ans == ",".join(o).lower().replace(" ", "").strip(" `,")
                    for o in order
                ]
            )
        )
    elif isinstance(order, list):
        order = ",".join(order)

    if "a-" in pred_ans:
        pred_ans = ",".join(re.findall("(?<=.-).", pred_ans))
    order = order.lower().replace(" ", "").strip(" `,*'")

    return int(pred_ans == order)


def bool_list_matching(pred: Dict, bool_str: list) -> int:
    pred_ans = (
        pred["answer"]
        .lower()
        .replace(" ", "")
        .replace("*", "")
        .replace("'", "")
        .strip(" `,")
    )
    if "[" in pred_ans:
        bool_str = "[" + ",".join(bool_str) + "]"
    else:
        bool_str = ",".join(bool_str)
    # print(bool_l, pred_ans)  # test
    return int(pred_ans == bool_str)


def number_matching(
    pred: Dict, value_to_match: Union[int, float]
) -> int:  # multi-question form has been added
    # extract number from pred_ans
    matches = re.findall(r"-?\d+(?:\.\d+)?", pred["answer"])
    result = matches[-1] if matches else None
    if result is None:
        return 0
    pred_ans = float(result)
    if isinstance(value_to_match, float):
        relative_error = abs(value_to_match) * 0.1
    else:
        relative_error = 1e-3
    return int(abs(pred_ans - value_to_match) < relative_error)


def get_score(gt: Dict, pred: Dict) -> Union[float, List[float]]:
    evaluator = gt["evaluator"]
    pred["raw_answer"] = pred["answer"]
    pred["answer"] = normalize_string(extract_answer(pred))
    registed_evaluator = set(
        [
            "key_items_matching",
            "choices_matching",
            "ordered_list_matching",
            "bool_list_matching",
            "number_matching",
            "location_matching",
            "interval_matching",
            "multi_interval_matching",
        ]
    )
    if evaluator not in registed_evaluator:
        raise ValueError(f"Unsupported evaluator: {evaluator}")
    return eval(evaluator)(pred, **gt["evaluator_kwargs"])


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    right: float = 0.0
    detailed_keys = ["task_category"]
    detailed_results = defaultdict(list)

    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        is_correct = get_score(gt, pred)
        if isinstance(is_correct, bool):
            is_correct_as_float = float(is_correct)
        else:
            is_correct_as_float = is_correct
        pred.update(gt)
        pred["correct"] = is_correct_as_float
        right += is_correct_as_float
        if detailed_keys:
            for key in detailed_keys:
                detailed_results[gt[key]].append(is_correct_as_float)
    results = {
        "accuracy": round(right / len(predictions) * 100, 2),
    }
    if detailed_keys:
        for key, values in detailed_results.items():
            results[key] = round(sum(values) / len(values) * 100, 2)
    return results
