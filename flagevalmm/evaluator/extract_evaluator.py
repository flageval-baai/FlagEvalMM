import re
from flagevalmm.models.api_response import ApiResponse
from flagevalmm.registry import EVALUATORS
from flagevalmm.evaluator import BaseEvaluator
import json
import os.path as osp
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple, Callable, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from flagevalmm.models import HttpClient
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)

### This prompts are modified from the MathVerse: https://github.com/ZrrSkywalker/MathVerse/blob/main/evaluation/prompts.py

demo_prompt_extract = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.
If the answer is a choice, you should output the choice letter rather than the answer.

1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)

2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D

3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)

4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null

5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3

6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1

7.
Model response: {answer}
Extracted answer: 
"""

demo_prompt_score = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.

[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0

[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0

[Question]: Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.A. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1 B. Can not determine.\n
[Standard Answer]: A
[Model_answer] : \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1
Judgement: 1

[Question]: {question}
[Standard Answer]: {answer}
[Model_answer] : {extracted_answer}
Judgement: """


### This prompts are modified from SimpleQA: https://cdn.openai.com/papers/simpleqa.pdf

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


@EVALUATORS.register_module()
class ExtractEvaluator(BaseEvaluator):
    """The evaluation method is implemented to utilize the llm to extract the answer from the model response.
    Two evaluation methods are supported:
    1. Extract + Compare: First extract answer from model response, then compare with ground truth
    2. SimpleQA: Directly grade the model response using SimpleQA grading template
    """

    def __init__(
        self,
        eval_model_name: str,
        use_llm_evaluator: bool = True,
        backend: str = "vllm",
        port: int = 8001,
        eval_func: Optional[Union[Callable, str]] = None,
        num_threads: int = 8,
        eval_method: str = "extract_compare",  # Can be "extract_compare" or "simpleqa"
        **kwargs,
    ) -> None:
        super().__init__(
            use_llm_evaluator=False,
            eval_func=eval_func,
            **kwargs,
        )

        self.use_llm_evaluator = use_llm_evaluator
        self.model_name = eval_model_name
        self.backend = backend
        self.port = port
        self.num_threads = num_threads
        self.eval_method = eval_method

        assert eval_method in [
            "extract_compare",
            "simpleqa",
        ], "eval_method must be either 'extract_compare' or 'simpleqa'"

        self.base_url = kwargs.pop(
            "base_url", "http://localhost:8000/v1/chat/completions"
        )
        if "localhost" in self.base_url:
            self.base_url = re.sub(
                r":(\d+)/",
                f":{self.port}/",
                self.base_url,
            )
        self.api_key = kwargs.pop("api_key", None)
        self.extra_args = kwargs.pop("extra_args", None)
        assert use_llm_evaluator, "use_llm_evaluator must be True"

        self.llm_evaluator = HttpClient(  # type: ignore
            model_name=self.model_name,
            url=self.base_url,
            api_key=self.api_key,
        )

    def extract_answer_by_llm(self, gt: Dict, pred: Dict):
        prompt = demo_prompt_extract.format(answer=pred["answer"])
        try:
            message = self.llm_evaluator.build_message(query=prompt)
            extracted_answer = self.llm_evaluator.infer(
                chat_messages=message, temperature=0, top_p=1, seed=42
            )
            # Handle both ApiResponse and string returns
            if hasattr(extracted_answer, "content"):
                content = extracted_answer.content
            else:
                content = str(extracted_answer)
            return content.replace("Extracted answer: ", "").strip()
        except Exception as e:
            logger.error(f"Error in evaluating by llm: {e}")
            return "[FAILED]"

    def compare_answer(self, gt: Dict, extracted_answer: str):
        string_compare = gt["answer"] == extracted_answer

        prompt = demo_prompt_score.format(
            question=gt["question"],
            answer=gt["answer"],
            extracted_answer=extracted_answer,
        )
        message = self.llm_evaluator.build_message(query=prompt)
        try:
            compare_result = self.llm_evaluator.infer(
                chat_messages=message, temperature=0, top_p=1, seed=42
            )
            assert isinstance(
                compare_result, ApiResponse
            ), f"response is not an ApiResponse: {compare_result}"
            content = compare_result.content
            return content.replace("Judgement: ", "").strip()

        except Exception as e:
            logger.error(f"Error in evaluating by llm: {e}")
            return False, string_compare

    def grade_by_simpleqa(self, gt: Dict, pred: Dict) -> Tuple[str, int]:
        """Grade the prediction using SimpleQA grading template"""
        prompt = GRADER_TEMPLATE.format(
            question=gt["question"],
            target=gt["answer"],
            predicted_answer=pred["answer"],
        )
        try:
            message = self.llm_evaluator.build_message(query=prompt)
            grade_letter_response = self.llm_evaluator.infer(
                chat_messages=message, temperature=0, top_p=1, seed=42
            )
            assert isinstance(
                grade_letter_response, ApiResponse
            ), f"response is not an ApiResponse: {grade_letter_response}"
            grade_letter = grade_letter_response.content.strip()

            # Convert grade letter to score
            if grade_letter == "A":  # CORRECT
                return "CORRECT", 1
            elif grade_letter == "B":  # INCORRECT
                return "INCORRECT", 0
            else:  # NOT_ATTEMPTED or invalid response
                return "NOT_ATTEMPTED", 0
        except Exception as e:
            logger.error(f"Error in SimpleQA grading: {e}")
            return "[FAILED]", 0

    def process_single_prediction(self, pred: Dict, gt: Dict) -> Tuple[Dict, int]:
        """Process a single prediction in a thread-safe manner"""
        pred_result = pred.copy()  # Create a copy to avoid thread safety issues
        pred_result["label"] = gt["answer"]

        if self.eval_method == "extract_compare":
            extracted_answer = self.extract_answer_by_llm(gt, pred)
            is_correct_by_llm = self.compare_answer(gt, extracted_answer)

            if is_correct_by_llm in ["0", "1"]:
                is_correct_by_llm = int(is_correct_by_llm)
            else:
                is_correct_by_llm = 0

            pred_result["extracted_answer"] = extracted_answer
            pred_result["correct"] = is_correct_by_llm
            pred_result["eval_method"] = "extract_compare"
            return pred_result, is_correct_by_llm
        else:  # simpleqa
            grade_result, is_correct = self.grade_by_simpleqa(gt, pred)
            pred_result["grade_result"] = grade_result
            pred_result["correct"] = is_correct
            pred_result["eval_method"] = "simpleqa"
            return pred_result, is_correct

    def cal_accuracy(
        self, annotations: Dict, predictions: List[Dict], *args, **kwargs
    ) -> Dict:
        right = 0
        processed_predictions = []

        # Create a thread pool
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks
            future_to_pred = {
                executor.submit(
                    self.process_single_prediction,
                    pred,
                    annotations[str(pred["question_id"])],
                ): pred
                for pred in predictions
            }

            for future in as_completed(future_to_pred):
                try:
                    pred_result, is_correct = future.result()
                    processed_predictions.append(pred_result)
                    right += is_correct
                except Exception as e:
                    logger.error(f"Error processing prediction: {e}")
                    # Add the original prediction to maintain order
                    pred = future_to_pred[future]
                    pred["extracted_answer"] = "[FAILED]"
                    pred["correct"] = 0
                    pred["label"] = annotations[str(pred["question_id"])]["answer"]
                    processed_predictions.append(pred)

        # Replace the original predictions with processed ones
        predictions.clear()
        predictions.extend(processed_predictions)

        results = {
            "accuracy": round(right / len(predictions) * 100, 2),
        }
        return results

    def process(self, dataset: Dataset, output_dir: str, **kwargs) -> Dict:
        """
        Args:
            dataset (Dataset): dataset instance
            output_dir: str
        """
        annotations = dataset.get_annotation()
        result_file = osp.join(output_dir, dataset.name + ".json")
        predictions = json.load(open(result_file))

        assert len(annotations) == len(predictions)
        results: Dict[str, Any] = {}
        predictions, filtered_predictions = self.filter_rejected(predictions, results)

        results.update(self.eval_func(annotations, predictions, self.llm_evaluator))

        self.save(results, predictions + filtered_predictions, dataset.name, output_dir)
        return results
