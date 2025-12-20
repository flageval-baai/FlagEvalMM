import json
import os.path as osp
from flagevalmm.registry import EVALUATORS
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)


@EVALUATORS.register_module()
class AggregationEvaluator:
    def __init__(self, evaluators, **kwargs) -> None:
        self.evaluators = evaluators

    def save_results(self, dataset_name, output_info, results, output_dir):
        json.dump(
            results, open(osp.join(output_dir, f"{dataset_name}_result.json"), "w")
        )
        # save evaluation results
        json.dump(
            output_info,
            open(osp.join(output_dir, f"{dataset_name}_evaluated.json"), "w"),
            ensure_ascii=False,
            indent=2,
        )

    def process(self, dataset, output_dir, **kwargs):
        """
        Args:
            dataset (Dataset): dataset instance
            answers (list): list of answers
        """
        annotations = dataset.get_annotation()
        dataset_name = dataset.name
        result_file = osp.join(output_dir, f"{dataset_name}.json")
        output_info = json.load(open(result_file))

        results = {}
        for evaluator_cfg in self.evaluators:
            logger.info(f"Processing {evaluator_cfg['type']}")
            evaluator = EVALUATORS.build(evaluator_cfg)
            results.update(
                evaluator.get_metric_results(
                    output_info=output_info,
                    output_dir=output_dir,
                    annotations=annotations,
                    dataset_name=dataset_name,
                )
            )
        self.save_results(dataset_name, output_info, results, output_dir)
        return results
