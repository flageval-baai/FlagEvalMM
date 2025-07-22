import json
from typing import Dict, List, Tuple

from flagevalmm.models import GPT
from collections import defaultdict

PROMPT_JSON_TEMPLATE = """
\nRemember, your output should only contain the following json format:
{
"answer":,
}
Be sure to use double backslashes if necessary, not single backslashe.
"""

def get_gpt4_extract_ICE():
    example_1 = """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)
"""  # 抽取元组格式

    example_2 = """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D
"""  # 抽取选择题选项

    example_3 = """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
"""  # 抽取包含多部分的复杂答案

    example_4 = """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null
"""  # 处理模型未能解答的情况

    example_5 = """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3
"""  # 抽取数值答案

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1
"""  # 抽取公式答案

    return [example_1, example_2, example_3, example_4, example_5, example_6]


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0
"""  # 括号不匹配，错误

    example_2 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0
"""  # 选项错误

    example_3 = """
[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0
"""  # 答案不完整，错误

    example_4 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0
"""  # 未给出答案，错误

    return [example_1, example_2, example_3, example_4]


def build_gpt4_score_prompt(question: str, standard_answer: str, model_answer: str) -> str:
    task_description = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm, \sqrt{10} and √10.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
"""
    demo_prompt = task_description
    examples = get_gpt4_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'

    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {standard_answer}
[Model_answer] : {model_answer}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}{PROMPT_JSON_TEMPLATE}'
    return full_prompt


def build_gpt4_extract_prompt(predictions: str) -> str:
    task_description = """
    I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
    """  # noqa
    prediction = str(predictions)
    examples = get_gpt4_extract_ICE()
    for example in examples:
        task_description += example + '\n\n'

    test_prompt = f"7.\nModel response: '{prediction}'\nExtracted Answer: "
    full_prompt = f'{task_description}{test_prompt}{PROMPT_JSON_TEMPLATE}'
    return full_prompt


def post_check_score(answer, label, prefetch=False):
    label = str(label).strip()
    answer = str(answer).strip()

    if answer == label:
        return answer if prefetch else True
    else:
        return False

def extract_answer_by_gpt(model_answer: str, model: GPT):
    prompt = build_gpt4_extract_prompt(model_answer)
    message = model.build_message(prompt)
    max_try = 5
    temperature = 0
    try_times = 0
    while try_times < max_try:
        try:
            response = model.infer(chat_messages=message, temperature=temperature)
            content = json.loads(response)
            return content['answer'] if 'answer' in content else {'answer': None}
        except Exception:
            try_times += 1
            temperature = try_times * 0.5
    return content



def get_score_by_gpt(question, answer, label, model: GPT) -> int:
    prompt = build_gpt4_score_prompt(question, label, answer)
    message = model.build_message(prompt)
    # 预检：如果字符串完全匹配，直接判为1分，节省API调用
    if post_check_score(answer, label, prefetch=True):
        return 1
    max_try = 5
    temperature = 0
    try_times = 0
    while try_times < max_try:
        try:
            response = model.infer(chat_messages=message, temperature=temperature)
            content = json.loads(response)
            answer = content['answer'] if 'answer' in content else {'answer': None}
            return int(answer)
        except Exception:
            try_times += 1
            temperature = try_times * 0.5
    return 0



def get_result(annotations: Dict, predictions: List[Dict], llm_evaluator: GPT) -> Dict:
    detailed_key = 'category'
    right = 0
    detailed_results = defaultdict(list)
    for i, pred in enumerate(predictions):
        question_id = pred["question_id"]
        gt = annotations[question_id]
        extract_answer = extract_answer_by_gpt(pred["reason"], llm_evaluator)
        predictions[i]["raw_answer"] = extract_answer
        score = get_score_by_gpt(pred["question"], extract_answer, gt['answer'], llm_evaluator)
        predictions[i]["answer"] = score
        is_correct = score == 1
        pred["correct"] = is_correct
        pred["label"] = gt["answer"]
        right += is_correct
        detailed_results[gt[detailed_key]].append(is_correct)
    results = {"accuracy": round(right / len(predictions) * 100, 2)}
    for key, values in detailed_results.items():
        results[key] = round(sum(values) / len(values) * 100, 2)
    return results