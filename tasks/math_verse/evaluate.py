import json
from copy import deepcopy
from typing import Dict, List

from flagevalmm.models import GPT
from collections import defaultdict

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
    is_correct = gt["answer"].upper() == pred["answer"]
    return is_correct


EXTRACT_AND_SCORING_PROMPT = """You will be given {NUMBER} question with ground truth answers and model responses under an overarching question. You need to go through each of the questions, extract the final answer from the model response, compare it with the ground truth answer, and then assign a binary score.
Avoid providing explanations in your response. Your response must be in json format, each item with keys ['id', 'extract_answer', 'score'] where the value for any `extract_answer` is your extracted answer and `score` is an interger in [0, 1] based on the following rules:\n

### RULES ###
- For multiple-choice questions, if the extracted answer matches the ground truth answer, assign a score of 1, otherwise assign a score of 0.
- For free-form questions, if the extracted answer has exactly the same meaning as the ground truth answer, assign a score of 1, otherwise assign a score of 0.

Here is an example:

### Example Start ###
Q1:
[Question]: Write the set of numbers represented on the number line in interval notation.
[Ground Truth]: (-2,1]
[Model Response] : Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)

Q2:
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Ground Truth]: C
[Model Response]: Answer is: C \u221a{{3}}

Q3:
[Question]: Find the domain and range of the function f using interval notation.
[Ground] Truth: domain: [-4, 0) and range: (-3, 1]
[Model Response]: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'

Q4:
[Question]: Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.
[Ground Truth]: \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1
[Model Response] : have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation is \\frac{{y^2}}{{9}} + \\frac{{x^2}}{{81}} = 1


Your output:
{{"results": [
{{
    "id": "Q1",
    "extract_answer": "(-2, 1)",
    "score": 0
}},
{{
    "id": "Q2",
    "extract_answer": "C",
    "score": 1
}},
{{
    "id": "Q3",
    "extract_answer": "Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)",
    "score": 0
}},
{{
    "id": "Q4",
    "extract_answer": "\\frac{{y^2}}{{9}} + \\frac{{x^2}}{{81}} = 1",
    "score": 1
}}
]
}}
### Example End ###
"""


def build_prompt(data_collection: List[Dict]) -> str:
    prompt = deepcopy(EXTRACT_AND_SCORING_PROMPT)
    prompt = prompt.format(NUMBER=len(data_collection))
    for i, data in enumerate(data_collection):
        prompt += f"\nQ{i + 1}:\n[Question]: {data['question']}\n[Ground Truth]: {data['gt']}\n[Model Response]: {data['model_response']}\n"
    return prompt


def get_score_by_gpt(data_collection: List, model: GPT) -> List[Dict]:
    prompt = build_prompt(data_collection)
    message = model.build_message(prompt)
    max_try = 4
    temperture = 0
    try_times = 0
    while try_times < max_try:
        try:
            response = model.infer(
                chat_messages=message, temperature=temperture, seed=42
            )
            content = json.loads(response.content)
            # check format
            assert isinstance(content, dict) and "results" in content, content
            assert len(content["results"]) == len(
                data_collection
            ), "results length mismatch"
            for cont in content["results"]:
                assert "id" in cont
                assert "extract_answer" in cont
                assert "score" in cont
            return content["results"]
        except Exception as e:
            print(try_times, e)
            temperture += 0.1
            try_times += 1
    # build dummy content
    content = []
    for i in range(len(data_collection)):
        content.append({"id": f"Q{i + 1}", "extract_answer": "", "score": 0})
    return content


def get_result(annotations: Dict, predictions: List[Dict], llm_evaluator: GPT) -> Dict:
    eval_batch_size = 5
    scores = defaultdict(lambda: {"accuracy": 0, "correct": 0, "total": 0})
    data_collection = []
    results = {}
    for i, pred in enumerate(predictions):
        question_id = pred["question_id"]
        gt = annotations[question_id]
        question_type = gt["question_type"]
        if question_type == "multi-choice":
            is_correct = evaluate_multiple_choice(gt, pred)
            pred["correct"] = is_correct
        else:
            data_collection.append(
                {
                    "question": gt["question"],
                    "gt": gt["answer"],
                    "model_response": pred["answer"],
                    "problem_version": gt["problem_version"],
                    "index": i,
                }
            )
            if len(data_collection) == eval_batch_size:
                response = get_score_by_gpt(data_collection, llm_evaluator)
                for j, res in enumerate(response):
                    index = data_collection[j]["index"]
                    predictions[index]["extract_answer"] = res["extract_answer"]
                    predictions[index]["correct"] = res["score"]
                data_collection = []
    if len(data_collection) > 0:
        response = get_score_by_gpt(data_collection, llm_evaluator)
        for j, res in enumerate(response):
            index = data_collection[j]["index"]
            predictions[index]["extract_answer"] = res["extract_answer"]
            predictions[index]["correct"] = res["score"]

    for pred in predictions:
        problem_version = annotations[pred["question_id"]]["problem_version"]
        pred["gt"] = annotations[pred["question_id"]]["answer"]
        scores[problem_version]["total"] += 1
        scores[problem_version]["correct"] += pred["correct"]
        scores["Total"]["total"] += 1
        scores["Total"]["correct"] += pred["correct"]
    # # calculate accuracy
    for key in scores:
        scores[key]["accuracy"] = round(
            scores[key]["correct"] / scores[key]["total"] * 100, 2
        )
    results.update(scores)
    results["accuracy"] = scores["Total"]["accuracy"]
    return results
