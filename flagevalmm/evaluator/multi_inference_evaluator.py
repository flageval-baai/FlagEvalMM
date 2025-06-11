import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Tuple, Union
import os.path as osp
from flagevalmm.evaluator.base_evaluator import BaseEvaluator
from flagevalmm.registry import EVALUATORS
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QuestionMapping:
    original_question_id: str
    is_multi_inference: bool
    inference_index: int
    total_inferences: int


@EVALUATORS.register_module()
class MultiInferenceEvaluator(BaseEvaluator):
    """
    Multi-inference evaluator for evaluating the accuracy of model's multiple inference results.
    """

    def __init__(
        self,
        is_clean: bool = True,
        use_llm_evaluator: bool = False,
        eval_func: Optional[Union[Callable, str]] = None,
        base_dir: str = "",
        detailed_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            is_clean=is_clean,
            use_llm_evaluator=use_llm_evaluator,
            eval_func=eval_func,
            base_dir=base_dir,
            detailed_keys=detailed_keys,
            **kwargs,
        )

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
            multiple_raw_answers = pred.get("multiple_raw_answers", [pred["answer"]])

            if len(multiple_raw_answers) == 1:
                # Single inference - keep as is
                expanded_predictions.append(pred)
                question_mapping[len(expanded_predictions) - 1] = QuestionMapping(
                    original_question_id=pred["question_id"],
                    is_multi_inference=False,
                    inference_index=0,
                    total_inferences=1,
                )
            else:
                print(multiple_raw_answers)
                # Multiple inferences - expand into separate predictions
                for idx, answer in multiple_raw_answers.items():
                    i = int(idx.split("_")[-1])
                    expanded_pred = pred.copy()
                    expanded_pred["answer"] = answer
                    expanded_pred["question_id"] = (
                        f"{pred['question_id']}_inference_{i}"
                    )
                    expanded_predictions.append(expanded_pred)

                    question_mapping[len(expanded_predictions) - 1] = QuestionMapping(
                        original_question_id=pred["question_id"],
                        is_multi_inference=True,
                        inference_index=i,
                        total_inferences=len(multiple_raw_answers),
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

    def process(self, dataset, output_dir, **kwargs):
        """
        Process dataset evaluation using BaseEvaluator's eval_func method.
        """
        annotation = dataset.get_annotation()
        dataset_name = dataset.name
        result_file = osp.join(output_dir, f"{dataset_name}.json")

        if not osp.exists(result_file):
            logger.error(f"Result file not found: {result_file}")
            return {}

        predictions = json.load(open(result_file))

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
            expanded_annotations[qid] = annotation[original_qid]

        # Step 3: Filter rejected predictions (following BaseEvaluator pattern)
        results = {}
        expanded_predictions, filtered_predictions = self.filter_rejected(
            expanded_predictions, results
        )

        # Step 4: Use eval_func to compute results (following BaseEvaluator pattern)
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

        total_score = sum(pred["correct"] for pred in aggregated_predictions)
        total_questions = len(aggregated_predictions)
        overall_accuracy = (
            round(total_score / total_questions * 100, 2)
            if total_questions > 0
            else 0.0
        )

        results.update(
            {
                "overall_accuracy": overall_accuracy,
                "total_questions": total_questions,
                "total_score": total_score,
                "single_inference_count": stats["single_inference_count"],
                "multi_inference_count": stats["multi_inference_count"],
            }
        )

        all_predictions = aggregated_predictions + filtered_predictions
        self.save(results, all_predictions, dataset_name, output_dir)

        logger.info(f"Overall Accuracy: {results.get('overall_accuracy', 0):.2f}%")

        return results
