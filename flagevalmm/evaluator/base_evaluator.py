import json
import re
import difflib
import pprint
import importlib.util
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Callable, Union, Any
from torch.utils.data import Dataset
import os.path as osp
from flagevalmm.registry import EVALUATORS
from flagevalmm.evaluator.pre_process import process_multiple_choice, normalize_string
from flagevalmm.models import GPT
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = """
You are an expert evaluator for visual question answering tasks. Your job is to assess the correctness of a model's answer compared to the ground truth (GT) answer. Follow these guidelines:

1. Carefully read the problem description, GT answer and the model's answer.
2. Extract the valid part of the model's answer, ignoring any irrelevant information.
3. Compare the extracted answer to the GT answer.
4. Assign a score:
   - 1 if the answer is correct or substantially equivalent to the GT.
   - 0 if the answer is incorrect or significantly different from the GT.
5. Be lenient with minor differences in phrasing or formatting.

Provide your evaluation in the following JSON format:
{{
  "extracted_answer": "The relevant part of the model's answer",
  "score": 0 or 1,
}}

Here are two examples:

Example 1:
Question: <image 1>What is the cat doing?
GT Answer: The cat is sitting on a red couch.
Model's Answer: Based on the image, I can see that the cat is lounging on a red sofa. The feline appears to be comfortable and relaxed in its position.

Evaluation:
{{
  "extracted_answer": "The cat is lounging on a red sofa",
  "score": 1,
}}

Example 2:
Question: <image 1> Calculate the area of the rectangle.
GT Answer: 4.2
Model's Answer: Answer: There area is: 4.1

Evaluation:
{{
  "extracted_answer": "4.1",
  "score": 0,
}}

Now, please evaluate the following answer:

Question: {question}
GT Answer: {gt_answer}
Model's Answer: {pred_answer}

Please provide your evaluation in the specified JSON format.
"""


@EVALUATORS.register_module()
class BaseEvaluator:
    def __init__(
        self,
        is_clean: bool = True,
        use_llm_evaluator: bool = False,
        eval_func: Optional[Union[Callable, str]] = None,
        base_dir: str = "",
        detailed_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.is_clean = is_clean
        self.base_dir = base_dir
        self.eval_func = self.get_eval_func(eval_func)
        self.use_llm_evaluator = use_llm_evaluator
        if use_llm_evaluator:
            self.llm_evaluator = GPT(
                model_name=kwargs.pop("model_name"),
                api_key=kwargs.pop("api_key"),
                base_url=kwargs.pop("base_url"),
                json_mode=True,
                **kwargs,
            )
        self.detailed_keys = detailed_keys

    def get_eval_func(self, eval_func: Optional[Union[Callable, str]]):
        if eval_func is None:
            return self.cal_accuracy
        if isinstance(eval_func, str):
            # Store the path for later loading
            self.eval_func_path = (
                eval_func
                if osp.isabs(eval_func)
                else osp.join(self.base_dir, eval_func)
            )
            return self._load_and_call_eval_func
        return eval_func

    def _load_and_call_eval_func(self, *args, **kwargs):
        # Load the module and call the function dynamically when needed
        spec = importlib.util.spec_from_file_location("evaluate", self.eval_func_path)
        if spec is None:
            raise ImportError(f"Could not load module from {self.eval_func_path}")
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError(f"Module {self.eval_func_path} has no loader")
        spec.loader.exec_module(module)
        return getattr(module, "get_result")(*args, **kwargs)

    def evaluate_multiple_choice(self, gt: Dict, pred: Dict) -> bool:
        pred["raw_answer"] = pred["answer"]
        pred["answer"] = self.maybe_clean_answer(pred["answer"])
        if len(pred["answer"]) > 1:
            pred["answer"] = pred["answer"][0]
        is_correct = bool(gt["answer"].upper() == pred["answer"])
        return is_correct

    def evaluate_fill_blank_by_rule(
        self, gt: Dict, pred: Dict, simality_threshold: float = 0.7
    ) -> Tuple[bool, str]:
        pred["raw_answer"] = pred["answer"]
        if "</think>" in pred["answer"]:
            pred["answer"] = pred["answer"].split("</think>")[1]
        splited_answer = pred["answer"].split("\n")
        cleaned_answers: List[str] = []
        for raw_answer in splited_answer:
            s = normalize_string(raw_answer)
            if s:
                cleaned_answers.append(s)

        gt_answer: str = normalize_string(gt["answer"])
        pred["answer"] = "\n".join(cleaned_answers)

        for cleaned_answer in cleaned_answers:
            simality = difflib.SequenceMatcher(
                None, str(cleaned_answer), str(gt_answer)
            ).ratio()
            if simality > simality_threshold:
                return True, cleaned_answer

        return False, "\n".join(cleaned_answers)

    def evaluate_multiple_response(self, gt: Dict, pred: Dict) -> Tuple[bool, str]:
        answer_str: str = self.maybe_clean_answer(pred["answer"])
        answer_matches: List[str] = re.findall("[ABCDEFGH]", answer_str)

        cleaned_answer = "".join(sorted(set(answer_matches)))
        pred["answer"] = cleaned_answer
        is_right = gt["answer"].upper() == cleaned_answer
        return is_right, cleaned_answer

    def verify_grading_output(self, data):
        if "extracted_answer" not in data:
            return False
        if "score" not in data:
            return False
        if data["score"] not in [0, 1]:
            return False
        return True

    def evaluate_by_llm(self, gt: Dict, pred: Dict) -> Tuple[bool, str]:
        prompt = PROMPT_TEMPLATE.format(
            question=gt["question"], gt_answer=gt["answer"], pred_answer=pred["answer"]
        )
        message = self.llm_evaluator.build_message(query=prompt)
        try:
            response = self.llm_evaluator.infer(
                chat_messages=message, temperature=0, top_p=1, seed=42
            )
            content = json.loads(response)
        except Exception as e:
            logger.error(f"Error in evaluating by llm: {e}")
            return False, "[FAILED]"
        # verify the integrity of the response
        if not self.verify_grading_output(content):
            logger.warning(f"grading output is not correct: {content}")
            return False, "[FAILED]"
        return content["score"], content["extracted_answer"]

    def cal_accuracy(
        self, annotations: Dict, predictions: List[Dict], *args, **kwargs
    ) -> Dict:
        right = 0
        detailed_results = defaultdict(list)
        for pred in predictions:
            question_id = str(pred["question_id"])
            gt = annotations[question_id]
            if gt["question_type"] == "multiple-choice":
                is_correct = self.evaluate_multiple_choice(gt, pred)
            else:
                if self.use_llm_evaluator:
                    is_correct, cleaned_answer = self.evaluate_by_llm(gt, pred)
                else:
                    is_correct, cleaned_answer = self.evaluate_fill_blank_by_rule(
                        gt, pred
                    )
                pred["raw_answer"] = pred["answer"]
                pred["answer"] = cleaned_answer
            pred["correct"] = is_correct
            pred["label"] = gt["answer"]
            right += is_correct
            if self.detailed_keys:
                for key in self.detailed_keys:
                    detailed_results[gt[key]].append(is_correct)
        results = {
            "accuracy": round(right / len(predictions) * 100, 2),
        }
        if self.detailed_keys:
            for key, values in detailed_results.items():
                results[key] = round(sum(values) / len(values) * 100, 2)
        return results

    def maybe_clean_answer(self, answer: str) -> str:
        if not self.is_clean:
            return answer
        if len(answer) == 1:
            return answer.upper()
        answer = process_multiple_choice(answer)
        return answer

    def filter_rejected(
        self, predictions: List[Dict], results: Dict
    ) -> Tuple[List[Dict], List[Dict]]:
        reject_keyword = [
            "Error code",
            "Can not answer because of",
            "Input data may contain inappropriate content",
        ]
        predictions_keeped = []
        predictions_filtered = []
        for pred in predictions:
            if any([pred["answer"].startswith(keyword) for keyword in reject_keyword]):
                pred["raw_answer"] = pred["answer"]
                predictions_filtered.append(pred)
            else:
                predictions_keeped.append(pred)
        filtered_number = len(predictions) - len(predictions_keeped)
        if filtered_number > 0:
            results["reject_info"] = {
                "reject_rate": round(filtered_number / len(predictions) * 100, 2),
                "reject_number": filtered_number,
                "total_question": len(predictions),
            }
        return predictions_keeped, predictions_filtered

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

        if self.use_llm_evaluator:
            results.update(self.eval_func(annotations, predictions, self.llm_evaluator))
        else:
            results.update(self.eval_func(annotations, predictions))

        self.save(results, predictions + filtered_predictions, dataset.name, output_dir)
        return results

    def save(
        self, results: Dict, answers: List[Dict], dataset_name: str, output_dir: str
    ):
        pprint.pprint(results)
        json.dump(
            results,
            open(osp.join(output_dir, f"{dataset_name}_result.json"), "w"),
            ensure_ascii=False,
            indent=2,
        )
        json.dump(
            answers,
            open(osp.join(output_dir, f"{dataset_name}_evaluated.json"), "w"),
            ensure_ascii=False,
            indent=2,
        )
