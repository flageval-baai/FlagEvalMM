import json
import re
import difflib
import pprint
import importlib.util
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Callable, Union, Any
from torch.utils.data import Dataset
import os.path as osp
from dataclasses import dataclass
from flagevalmm.models.api_response import ApiResponse
from flagevalmm.registry import EVALUATORS
from flagevalmm.evaluator.pre_process import process_multiple_choice, normalize_string
from flagevalmm.models import GPT
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = """
Below is a question with two corresponding question-answer pairs. The first pair is the ground truth: a question paired with its standard answer. The second pair is the question paired with the answer extracted from a model's output. Your task is to determine whether these two pairs form equivalent propositions—that is, whether the extracted pair completely and equivalently expresses the same meaning as the ground truth pair. For non-multiple-choice questions, if differences exist in the expression—for example, differences in measurement units (e.g., "0.5m" vs. "50cm"), variations in the spelling of people's names, or differences in the specificity of nouns—but the underlying meaning remains mathematically or logically equivalent, they should be considered equivalent. If the two pairs are equivalent, output "Judgement: Yes"; if not, output "Judgement: No". Explain your reasoning in detail.

[Question]: What is the most commonly used material that the blue item in the person's hand is made of?
[Standard Answer]: plastic
[Model_answer] : The blue item in the person's hand appears to be a frisbee, which is typically made of plastic.
Judgement: Yes

[Question]: In which year was the first of this animal born at Beijing Zoo was born?
[Standard Answer]: 1957
[Model_answer] : The first panda born at Beijing Zoo was in 1957.
Judgement: Yes

[Question]: Which country's airline does the plane in the image belong to?
[Standard Answer]: Australia
[Model_answer] : The plane in the image belongs to Qantas, an airline based in Australia.
Judgement: Yes

[Question]: Is it possible that the animal carries the gene for long fur?
[Standard Answer]: Yes
[Model_answer] : it's less likely to carry a gene for long fur unless it belongs to one of those specific breeds.
Judgement: No

[Question]: {question}
[Standard Answer]: {gt_answer}
[Model_answer] : {pred_answer}
Your response must end with: Judgement: Yes or Judgement: No
"""


@dataclass
class QuestionMapping:
    original_question_id: str
    is_multi_inference: bool
    inference_index: int
    total_inferences: int


@EVALUATORS.register_module()
class BaseEvaluator:
    def __init__(
        self,
        is_clean: bool = True,
        use_llm_evaluator: bool = False,
        eval_func: Optional[Union[Callable, str]] = None,
        base_dir: str = "",
        detailed_keys: Optional[List[str]] = None,
        aggregation_fields: Optional[List[str]] = ["raw_answer"],
        **kwargs,
    ) -> None:
        self.is_clean = is_clean
        self.base_dir = base_dir
        self.eval_func = self.get_eval_func(eval_func)
        self.use_llm_evaluator = use_llm_evaluator
        if use_llm_evaluator:

            self.llm_evaluator = GPT(
                model_name=kwargs.pop("eval_model_name"),
                api_key=kwargs.pop("api_key"),
                base_url=kwargs.pop("base_url"),
                use_cache=kwargs.pop("use_cache", True),
                **kwargs,
            )
        self.detailed_keys = detailed_keys
        self.aggregation_fields = aggregation_fields or []

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

    def statistics_tokens(self, predictions: List[Dict]) -> Dict:
        average_tokens = 0.0
        average_prompt_tokens = 0.0
        average_completion_tokens = 0.0
        for pred in predictions:
            if not pred.get("usage"):
                continue
            average_tokens += pred["usage"]["total_tokens"]
            average_prompt_tokens += pred["usage"]["prompt_tokens"]
            average_completion_tokens += pred["usage"]["completion_tokens"]
        average_tokens = average_tokens / len(predictions)
        average_prompt_tokens = average_prompt_tokens / len(predictions)
        average_completion_tokens = average_completion_tokens / len(predictions)
        return {
            "average_tokens": average_tokens,
            "average_prompt_tokens": average_prompt_tokens,
            "average_completion_tokens": average_completion_tokens,
        }

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

    def expand_multi_inference_predictions(
        self, predictions: List[Dict]
    ) -> Tuple[List[Dict], Dict[int, QuestionMapping]]:
        """
        Expand multiple inference predictions into individual predictions.

        Returns:
            expanded_predictions: List of individual predictions
            question_mapping: Mapping from expanded prediction index to original question info
        """
        expanded_predictions = []
        question_mapping = {}

        for pred in predictions:
            # Check if answer is a dictionary (multiple inferences) or string (single inference)
            answer = pred["answer"]

            if isinstance(answer, dict):
                # Multiple inferences - expand into separate predictions
                for key, single_answer in answer.items():
                    i = int(key.split("_")[-1])
                    expanded_pred = pred.copy()
                    expanded_pred["answer"] = single_answer
                    expanded_pred["question_id"] = (
                        f"{pred['question_id']}_inference_{i}"
                    )
                    expanded_predictions.append(expanded_pred)

                    question_mapping[len(expanded_predictions) - 1] = QuestionMapping(
                        original_question_id=pred["question_id"],
                        is_multi_inference=True,
                        inference_index=i,
                        total_inferences=len(answer),
                    )
            else:
                # Single inference - answer is a string
                expanded_predictions.append(pred)
                question_mapping[len(expanded_predictions) - 1] = QuestionMapping(
                    original_question_id=pred["question_id"],
                    is_multi_inference=False,
                    inference_index=0,
                    total_inferences=1,
                )

        return expanded_predictions, question_mapping

    def aggregate_multi_inference_results(
        self,
        expanded_predictions: List[Dict],
        question_mapping: Dict[int, QuestionMapping],
    ) -> Tuple[List[Dict], Dict]:
        """
        Aggregate results from expanded predictions back to original questions.

        Returns:
            aggregated_predictions: List of predictions with aggregated results
            stats: Statistics about the evaluation
        """
        # Group results by original question ID
        question_results = defaultdict(list)

        for i, pred in enumerate(expanded_predictions):
            mapping = question_mapping[i]
            original_qid = mapping.original_question_id

            question_results[original_qid].append(
                {
                    "inference_index": mapping.inference_index,
                    "correct": pred["correct"],
                    "answer": pred["answer"],
                    "is_multi_inference": mapping.is_multi_inference,
                    "total_inferences": mapping.total_inferences,
                    "expanded_pred": pred,
                }
            )

        # Aggregate results
        aggregated_predictions = []
        single_inference_count = 0
        multi_inference_count = 0

        for original_qid, results in question_results.items():
            results.sort(key=lambda x: x["inference_index"])  # Ensure correct order

            if results[0]["is_multi_inference"]:
                # Multiple inference case - take average
                inference_scores = [r["correct"] for r in results]
                average_accuracy = sum(inference_scores) / len(inference_scores)

                # Create aggregated prediction
                base_pred = results[0]["expanded_pred"].copy()
                base_pred["question_id"] = original_qid
                base_pred["correct"] = average_accuracy
                base_pred["answer"] = {
                    f"inference_{idx}": r["answer"] for idx, r in enumerate(results)
                }
                base_pred["inference_scores"] = inference_scores
                base_pred["num_inferences"] = len(results)

                # Handle aggregation fields
                for field in self.aggregation_fields:
                    if field in base_pred:
                        # Aggregate the field values from all inferences
                        field_values = []
                        for r in results:
                            if field in r["expanded_pred"]:
                                field_values.append(r["expanded_pred"][field])
                        if field_values:
                            base_pred[field] = field_values

                aggregated_predictions.append(base_pred)
                multi_inference_count += 1
            else:
                # Single inference case
                base_pred = results[0]["expanded_pred"].copy()
                base_pred["question_id"] = original_qid
                aggregated_predictions.append(base_pred)
                single_inference_count += 1

        stats = {
            "single_inference_count": single_inference_count,
            "multi_inference_count": multi_inference_count,
            "total_questions": len(question_results),
        }

        return aggregated_predictions, stats

    def has_multi_inference(self, predictions: List[Dict]) -> bool:
        """
        Check if any prediction contains multiple inference results.
        """
        for pred in predictions:
            if isinstance(pred["answer"], dict):
                return True
        return False

    def evaluate_multiple_choice(self, gt: Dict, pred: Dict) -> bool:
        if not isinstance(pred["answer"], str):
            return False
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

    def extract_judgement_result(self, response_text: str) -> Tuple[bool, str]:
        """
        Extract judgement result from LLM response using regex.
        Validates format and extracts the result in one step.
        Returns: (is_correct, extracted_response)
        """

        # Extract the judgement using regex
        judgement_pattern = r"Judgement:\s*(Yes|No)"
        match = re.search(judgement_pattern, response_text, re.IGNORECASE)

        if match:
            judgement = match.group(1).lower()
            is_correct = judgement == "yes"
            # Return the full response as extracted answer for logging purposes
            return is_correct, response_text.strip()
        else:
            logger.warning(
                f"Could not extract judgement from response: {response_text}"
            )
            return False, "[FAILED]"

    def evaluate_by_llm(self, gt: Dict, pred: Dict) -> Tuple[bool, str]:
        prompt = PROMPT_TEMPLATE.format(
            question=gt["question"], gt_answer=gt["answer"], pred_answer=pred["answer"]
        )
        message = self.llm_evaluator.build_message(query=prompt)
        try:
            response = self.llm_evaluator.infer(
                chat_messages=message, temperature=0, top_p=1, seed=42
            )
            assert isinstance(
                response, ApiResponse
            ), f"response is not an ApiResponse: {response}"

            # Get the raw text response instead of parsing as JSON
            response_text = response.content

        except Exception as e:
            logger.error(f"Error in evaluating by llm: {e}")
            return False, "[FAILED]"

        # Extract judgement result using regex (validates format and extracts in one step)
        is_correct, extracted_response = self.extract_judgement_result(response_text)
        return is_correct, extracted_response

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
            elif gt["question_type"] == "llm-judge" and self.use_llm_evaluator:
                is_correct, judgement_result = self.evaluate_by_llm(gt, pred)
                pred["judgement_result"] = judgement_result
            elif gt["question_type"] == "fill-blank":
                is_correct, cleaned_answer = self.evaluate_fill_blank_by_rule(gt, pred)
                pred["answer"] = cleaned_answer
            else:
                raise ValueError(f"Unsupported question type: {gt['question_type']}")
            pred["raw_answer"] = pred["answer"]
            pred["correct"] = is_correct
            pred["label"] = gt["answer"]
            pred["question_type"] = gt["question_type"]
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
            # Handle both string and dictionary formats for pred["answer"]
            should_reject = False

            if isinstance(pred["answer"], str):
                # Single answer case (no num-infer)
                should_reject = any(
                    [pred["answer"].startswith(keyword) for keyword in reject_keyword]
                )
            elif isinstance(pred["answer"], dict):
                # Multiple inference case (with num-infer)
                # Check if any of the inference results starts with reject keywords
                should_reject = all(
                    [
                        inference_result.startswith(keyword)
                        for inference_result in pred["answer"].values()
                        if isinstance(inference_result, str)
                        for keyword in reject_keyword
                    ]
                )

            if should_reject:
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

        if not osp.exists(result_file):
            logger.error(f"Result file not found: {result_file}")
            return {}

        predictions = json.load(open(result_file))

        # Check if we have multi-inference predictions
        if self.has_multi_inference(predictions):
            logger.info(
                "Detected multi-inference predictions, using multi-inference evaluation"
            )
            return self._process_multi_inference(
                dataset, predictions, annotations, output_dir
            )
        else:
            logger.info(
                "Single inference predictions detected, using standard evaluation"
            )
            return self._process_single_inference(
                dataset, predictions, annotations, output_dir
            )

    def _process_single_inference(
        self, dataset, predictions: List[Dict], annotations: Dict, output_dir: str
    ) -> Dict:
        """Process single inference predictions (original BaseEvaluator logic)"""
        assert len(annotations) == len(predictions)
        results: Dict[str, Any] = {}
        predictions, filtered_predictions = self.filter_rejected(predictions, results)

        if self.use_llm_evaluator:
            results.update(self.eval_func(annotations, predictions, self.llm_evaluator))
        else:
            results.update(self.eval_func(annotations, predictions))

        results.update(self.statistics_tokens(predictions))

        self.save(results, predictions + filtered_predictions, dataset.name, output_dir)
        return results

    def _process_multi_inference(
        self, dataset, predictions: List[Dict], annotations: Dict, output_dir: str
    ) -> Dict:
        """Process multi-inference predictions"""
        # Step 1: Expand multiple inference predictions
        expanded_predictions, question_mapping = (
            self.expand_multi_inference_predictions(predictions)
        )
        logger.info(
            f"Expanded {len(predictions)} predictions to {len(expanded_predictions)} individual evaluations"
        )

        # Step 2: Create annotation mapping for expanded predictions
        expanded_annotations = {}
        for pred in expanded_predictions:
            # Extract original question ID from expanded question ID
            qid = pred["question_id"]
            if "_inference_" in qid:
                original_qid = qid.split("_inference_")[0]
            else:
                original_qid = qid
            expanded_annotations[qid] = annotations[original_qid]

        # Step 3: Filter rejected predictions
        results: Dict[str, Any] = {}
        expanded_predictions, filtered_predictions = self.filter_rejected(
            expanded_predictions, results
        )

        # Step 4: Use eval_func to compute results
        if self.use_llm_evaluator:
            base_results = self.eval_func(
                expanded_annotations, expanded_predictions, self.llm_evaluator
            )
        else:
            base_results = self.eval_func(expanded_annotations, expanded_predictions)

        # Step 5: Aggregate results back to original questions
        aggregated_predictions, stats = self.aggregate_multi_inference_results(
            expanded_predictions, question_mapping
        )

        results.update(base_results)
        logger.info(f"Multi-inference stats: {stats}")

        all_predictions = aggregated_predictions + filtered_predictions
        self.save(results, all_predictions, dataset.name, output_dir)

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
        answers = sorted(answers, key=lambda x: x.get("question_id", ""))
        json.dump(
            answers,
            open(osp.join(output_dir, f"{dataset_name}_evaluated.json"), "w"),
            ensure_ascii=False,
            indent=2,
        )
