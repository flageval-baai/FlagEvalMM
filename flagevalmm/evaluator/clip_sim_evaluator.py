import torch
import os.path as osp
import json
import numpy as np
from PIL import Image
from flagevalmm.registry import EVALUATORS
from flagevalmm.common.video_utils import read_video_pyav
from tqdm import tqdm

from torchmetrics.multimodal.clip_score import CLIPScore


@EVALUATORS.register_module()
class CLIPSIMEvaluator:
    def __init__(self, model, max_num_frames: int = 16, **kwargs) -> None:
        self.metric = CLIPScore(model_name_or_path=model).to("cuda")
        self.name = "clipsim"
        self.max_num_frames = max_num_frames

    def get_metric_results(self, output_info, output_dir, annotations, **kwargs):
        score_sum = 0
        num = 0
        for info in tqdm(output_info):
            video_path = osp.join(output_dir, info["video"])
            images = read_video_pyav(
                video_path=video_path,
                max_num_frames=self.max_num_frames,
                return_tensors=False,
            )
            question_id = info["id"]
            prompt = annotations[question_id]["prompt"]
            for image in images:
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to("cuda")
                clip_score = self.metric(image, prompt).item()
                score_sum += clip_score
                num += 1
            info["clipsim"] = clip_score
        return {"clipsim": score_sum / num}

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
        results["clipsim"] = self.get_metric_results(
            output_info=output_info, output_dir=output_dir, annotations=annotations
        )
        self.save_results(
            dataset_name=dataset_name,
            output_info=output_info,
            results=results,
            output_dir=output_dir,
        )
        return results
