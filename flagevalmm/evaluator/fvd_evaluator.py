import json
from typing import Optional
import os.path as osp
from flagevalmm.dataset.utils import get_data_root
from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR
import torch
import os
import torch.nn.functional as F
from flagevalmm.common.video_utils import read_video_pyav
from flagevalmm.registry import EVALUATORS
from huggingface_hub import hf_hub_download
import numpy as np
import math
from scipy.linalg import sqrtm
from tqdm import tqdm


def compute_stats(feats: np.ndarray):
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]
    return mu, sigma


def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)


def preprocess_single(video, resolution=224, sequence_length=None):
    video = video.permute(1, 0, 2, 3)
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start : h_start + resolution, w_start : w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = F.normalize(video, p=1, dim=1)
    video = (video - 0.5) * 2
    # print(video)

    return video.contiguous()


@EVALUATORS.register_module()
class FVDEvaluator:
    def __init__(
        self,
        model_path="i3d_torchscript.pt",
        example_dir: Optional[str] = None,
        base_dir: Optional[str] = None,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if not os.path.exists(model_path):
            model_path = hf_hub_download(
                repo_id=model_path, filename="i3d_torchscript.pt"
            )
        # print(filepath)
        i3d = torch.jit.load(model_path).eval().to("cuda")
        i3d = torch.nn.DataParallel(i3d)
        self.i3d = i3d

        if example_dir is None:
            data_root = get_data_root(
                data_root=None,
                config=config,
                cache_dir=FLAGEVALMM_DATASETS_CACHE_DIR,
                base_dir=base_dir,
            )
            self.example_dir = osp.join(data_root, "video")
        else:
            self.example_dir = example_dir

    def get_feats(self, video):
        # videos : torch.tensor BCTHW [0, 1]
        # detector_kwargs = dict(
        #     rescale=False, resize=False, return_features=True
        # )  # Return raw features before the softmax layer.
        feats = np.empty((0, 400))
        device = torch.device("cuda")
        video = preprocess_single(video).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = np.vstack(
                [
                    feats,
                    self.i3d(x=video, rescale=False, resize=False, return_features=True)
                    .detach()
                    .cpu()
                    .numpy(),
                ]
            )
        return feats

    def get_metric_results(self, output_info, output_dir, **kwargs):
        feats_fake = []
        feats_real = []

        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"]
        for root, _, files in tqdm(os.walk(self.example_dir)):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(root, file)
                    images = read_video_pyav(
                        video_path=video_path,
                        max_num_frames=16,
                        return_tensors=True,
                    )
                    feats_real.append(self.get_feats(images))

        for info in tqdm(output_info):
            video_path = osp.join(output_dir, info["video_path"])
            images = read_video_pyav(
                video_path=video_path,
                max_num_frames=16,
                return_tensors=True,
            )
            feats_fake.append(self.get_feats(images))

        fvd = frechet_distance(np.concatenate(feats_fake), np.concatenate(feats_real))
        return {"FVD": fvd}

    def process(self, dataset, output_dir, **kwargs):
        """
        Args:
            dataset (Dataset): dataset instance
            answers (list): list of answers
        """
        dataset_name = dataset.name
        result_file = osp.join(output_dir, f"{dataset_name}.json")
        output_info = json.load(open(result_file))

        results = self.get_metric_results(
            output_dir=output_dir, output_info=output_info
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
