from flagevalmm.registry import EVALUATORS
import os.path as osp
import json
import torch
from transformers import AutoModelForCausalLM
from PIL import Image
from collections import defaultdict
from typing import List, Dict
from torch.utils.data import Dataset


@EVALUATORS.register_module()
class OneAlignEvaluator:
    """
    This evaluator implements the image scoring methodology from the OneAlign project.
    For more details, see:
    - Paper: "Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels" (https://arxiv.org/abs/2312.17090)
    - Project: https://github.com/Q-Future/Q-Align
    """

    def __init__(
        self, model: str, metrics: List[str] = ["quality", "aesthetics"], **kwargs
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.metrics = metrics

    def get_metric_results(
        self, output_info: List[Dict], output_dir: str, **kwargs
    ) -> Dict:
        metric_results = defaultdict(list)
        for info in output_info:
            image_path = osp.join(output_dir, info["image"])
            image = Image.open(image_path)
            for metric in self.metrics:
                score = self.model.score([image], task_=metric, input_="image").item()
                info[f"one_align_{metric}"] = score
                metric_results[f"one_align_{metric}"].append(score)

        results = {}
        for k, v in metric_results.items():
            results[k] = round(sum(v) / len(v), 4)
        return results

    def process(self, dataset: Dataset, output_dir: str, **kwargs) -> Dict[str, float]:
        dataset_name = dataset.name
        result_file = osp.join(output_dir, f"{dataset_name}.json")
        output_info = json.load(open(result_file))

        results = self.get_metric_results(
            output_info=output_info, output_dir=output_dir
        )
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
        return results
