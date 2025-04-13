import os
import re
from collections import defaultdict
import signal
from flagevalmm.registry import EVALUATORS
from flagevalmm.evaluator import BaseEvaluator
import json
import re
from collections import defaultdict
import os.path as osp
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple, Callable, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

from flagevalmm.server.utils import get_random_port
from flagevalmm.registry import EVALUATORS
from flagevalmm.models import HttpClient
from flagevalmm.common.logger import get_logger
from flagevalmm.server.model_server import ModelServer

logger = get_logger(__name__)

### This prompts are from the MathVerse: https://github.com/ZrrSkywalker/MathVerse/blob/main/evaluation/prompts.py

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


@EVALUATORS.register_module()
class ExtractEvaluator(BaseEvaluator):
    """
    The evaluation method is implemented to utilize the llm to extract the answer from the model response.
    """
    def __init__(
        self,
        eval_model_name: str,
        use_llm_evaluator: bool = True,
        backend: str = "vllm",
        port: int = 8001,
        eval_func: Optional[Union[Callable, str]] = None,
        num_threads: int = 8,  # New parameter for controlling number of threads
        **kwargs,
    ) -> None:
        self.use_llm_evaluator = use_llm_evaluator
        self.model_name = eval_model_name
        self.backend = backend
        self.port = port
        self.eval_func = self.get_eval_func(eval_func)
        self.num_threads = num_threads
        # replace port in url

        self.base_url = kwargs.pop("base_url", "http://localhost:8000/v1/chat/completions")
        if "localhost" in self.base_url:
            self.base_url = re.sub(
                r":(\d+)/",
                f":{self.port}/",
                self.base_url,
            )
        self.api_key = kwargs.pop("api_key", None)
        self.extra_args = kwargs.pop("extra_args", None)
        assert use_llm_evaluator, "use_llm_evaluator must be True"

        self.llm_evaluator = HttpClient(
            model_name=self.model_name,
            url=self.base_url,
            api_key=self.api_key,
        )

    def extract_answer_by_llm(self, gt: Dict, pred: Dict) -> str:
        prompt = demo_prompt_extract.format(
            answer=pred["answer"]
        )
        try:
            message = self.llm_evaluator.build_message(query=prompt)
            extracted_answer = self.llm_evaluator.infer(
                chat_messages=message, temperature=0, top_p=1, seed=42
            )
            return extracted_answer.replace("Extracted answer: ", "").strip()
        except Exception as e:
            logger.error(f"Error in evaluating by llm: {e}")
            return "[FAILED]"
    
    def compare_answer(self, gt: Dict, extracted_answer: str):
        string_compare = gt["answer"] == extracted_answer

        prompt = demo_prompt_score.format(
            question=gt["question"], answer=gt["answer"], extracted_answer=extracted_answer
        )
        message = self.llm_evaluator.build_message(query=prompt)
        try:
            compare_result = self.llm_evaluator.infer(
                chat_messages=message, temperature=0, top_p=1, seed=42
            )
            return compare_result.replace("Judgement: ", "").strip()

        except Exception as e:
            logger.error(f"Error in evaluating by llm: {e}")
            return False, string_compare

    def process_single_prediction(self, pred: Dict, gt: Dict) -> Tuple[Dict, int]:
        """Process a single prediction in a thread-safe manner"""
        extracted_answer = self.extract_answer_by_llm(gt, pred)
        is_correct_by_llm = self.compare_answer(gt, extracted_answer)
        
        if is_correct_by_llm in ['0', '1']:
            is_correct_by_llm = int(is_correct_by_llm)
        else:
            is_correct_by_llm = 0
            
        pred_result = pred.copy()  # Create a copy to avoid thread safety issues
        pred_result["extracted_answer"] = extracted_answer
        pred_result["correct"] = is_correct_by_llm
        pred_result["label"] = gt["answer"]
        
        return pred_result, is_correct_by_llm

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
                    annotations[str(pred["question_id"])]
                ): pred
                for pred in predictions
            }
            
            # Process results as they complete with progress bar
            with tqdm(total=len(predictions), desc="Processing predictions") as pbar:
                for future in as_completed(future_to_pred):
                    try:
                        pred_result, is_correct = future.result()
                        processed_predictions.append(pred_result)
                        right += is_correct
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing prediction: {e}")
                        # Add the original prediction to maintain order
                        pred = future_to_pred[future]
                        pred["extracted_answer"] = "[FAILED]"
                        pred["correct"] = 0
                        pred["label"] = annotations[str(pred["question_id"])]["answer"]
                        processed_predictions.append(pred)
                        pbar.update(1)
        
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

