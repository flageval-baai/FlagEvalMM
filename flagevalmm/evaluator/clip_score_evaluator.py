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
class CLIPScoreEvaluator:
    def __init__(self, model_name_or_path, **kwargs) -> None:
        self.metric = CLIPScore(model_name_or_path=model_name_or_path).to("cuda")
        self.name = "clip_score"
        if "max_num_frames" in kwargs:
            self.max_num_frames = kwargs["max_num_frames"]

    def get_metric_results(self, output_info, output_dir, annotations, **kwargs):
        score_sum = 0
        assert (
            "image" in output_info[0] or "video" in output_info[0]
        ), "must be image or video!"
        if "image" in output_info[0]:
            for info in output_info:
                image_path = osp.join(output_dir, info["image"])
                image = Image.open(image_path).convert("RGB")
                image = np.array(image) * 255
                # to tensor
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to("cuda")
                question_id = info["id"]
                if isinstance(list(annotations.keys())[0], int):
                    question_id = int(question_id)
                prompt = annotations[question_id]["prompt"]
                clip_score = self.metric(image, prompt).item()
                score_sum += clip_score
                info["clip_score"] = clip_score
            result = {"clip_score": score_sum / len(output_info)}
        elif "video" in output_info[0]:
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
                    image = (
                        torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to("cuda")
                    )
                    clip_score = self.metric(image, prompt).item()
                    score_sum += clip_score
                    num += 1
                info["clipsim"] = clip_score
            result = {"clipsim": score_sum / num}
        return result

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
        results["clip_score"] = self.get_metric_results(
            output_info=output_info, output_dir=output_dir, annotations=annotations
        )
        self.save_results(
            dataset_name=dataset_name,
            output_info=output_info,
            results=results,
            output_dir=output_dir,
        )
        return results
