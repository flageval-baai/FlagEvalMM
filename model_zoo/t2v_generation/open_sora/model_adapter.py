import argparse
import torch
import json
import os
from retrying import retry
import requests
import os.path as osp
import colossalai
import torch.distributed as dist
from mmengine.runner import set_random_seed
from mmengine.config import Config

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.misc import to_torch_dtype
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from colossalai.cluster import DistCoordinator


def parse_args():
    parser = argparse.ArgumentParser(description="Model Adapter")
    parser.add_argument("--task", type=str, default="v2i")
    parser.add_argument("--server_ip", type=str, default="http://localhost")
    parser.add_argument("--server_port", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=1000)

    return parser.parse_args()


class ModelAdapter:
    def __init__(self, task, server_ip, server_port, timeout=1000):
        self.task = task
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        io_info = self.get_io_info()
        self.output_dir = io_info["output_dir"]
        self.meta_info = self.get_meta()

        # ======================================================
        # 1. cfg and init distributed env
        # ======================================================
        current_directory = osp.dirname(osp.realpath(__file__))
        cfg = Config.fromfile(osp.join(current_directory, "config.py"))
        if "multi_resolution" not in cfg:
            cfg["multi_resolution"] = False
        cfg.model["from_pretrained"] = io_info["checkpoint_path"]
        self.cfg = cfg
        print(self.cfg)

        # init distributed
        colossalai.launch_from_torch({})
        self.coordinator = DistCoordinator()

        if self.coordinator.world_size > 1:
            set_sequence_parallel_group(dist.group.WORLD)
            enable_sequence_parallelism = True
        else:
            enable_sequence_parallelism = False

        # ======================================================
        # 2. runtime variables
        # ======================================================
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = to_torch_dtype(self.cfg.dtype)
        set_random_seed(seed=self.cfg.seed)
        self.device = device
        # ======================================================
        # 3. build model & load weights
        # ======================================================
        # 3.1. build model
        input_size = (self.cfg.num_frames, *self.cfg.image_size)
        vae = build_module(self.cfg.vae, MODELS)
        self.latent_size = vae.get_latent_size(input_size)
        self.text_encoder = build_module(
            self.cfg.text_encoder, MODELS, device=device
        )  # T5 must be fp32
        model = build_module(
            self.cfg.model,
            MODELS,
            input_size=self.latent_size,
            in_channels=vae.out_channels,
            caption_channels=self.text_encoder.output_dim,
            model_max_length=self.text_encoder.model_max_length,
            dtype=dtype,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        self.text_encoder.y_embedder = (
            model.y_embedder
        )  # hack for classifier-free guidance

        # 3.2. move to device & eval
        self.vae = vae.to(device, dtype).eval()
        self.model = model.to(device, dtype).eval()

        # 3.3. build scheduler
        self.scheduler = build_module(self.cfg.scheduler, SCHEDULERS)

        # 3.4. support for multi-resolution
        self.model_args = dict()
        if self.cfg.multi_resolution:
            image_size = self.cfg.image_size
            hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(
                self.cfg.batch_size, 1
            )
            ar = torch.tensor(
                [[image_size[0] / image_size[1]]], device=device, dtype=dtype
            ).repeat(self.cfg.batch_size, 1)
            self.model_args["data_info"] = dict(ar=ar, hw=hw)

        self.dtype = dtype

    @retry(stop_max_attempt_number=5, wait_fixed=500)
    def get_meta(self):
        url = f"{self.server_ip}:{self.server_port}/meta_info"
        meta_info = requests.get(url, timeout=self.timeout).json()
        return meta_info

    @retry(stop_max_attempt_number=5, wait_fixed=500)
    def get_io_info(self):
        url = f"{self.server_ip}:{self.server_port}/io_info"
        io_info = requests.get(url, timeout=self.timeout).json()
        return io_info

    def infer(self, prompt):
        save_dir = self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        batch_prompts = [prompt]
        samples = self.scheduler.sample(
            self.model,
            self.text_encoder,
            z_size=(self.vae.out_channels, *self.latent_size),
            prompts=batch_prompts,
            device=self.device,
            additional_args=self.model_args,
        )
        samples = self.vae.decode(samples.to(self.dtype))
        return samples

    def run(self):
        url_template = f"{self.server_ip}:{self.server_port}/get_data?index={{}}"
        text_num = self.meta_info["length"]
        output_info = []
        for i in range(text_num):
            url = url_template.format(i)
            response = requests.get(url, timeout=500).json()
            prompt, question_id = response["prompt"], response["id"]
            samples = self.infer(prompt)

            video_out_name = f"{question_id}"

            if self.coordinator.is_master():
                for idx, sample in enumerate(samples):
                    print(f"Prompt: {prompt}")
                    save_path = os.path.join(self.output_dir, video_out_name)
                    save_sample(sample, fps=self.cfg.fps, save_path=save_path)
            output_info.append(
                {
                    "prompt": prompt,
                    "id": question_id,
                    "video_path": video_out_name + ".mp4",
                }
            )

        json.dump(
            output_info,
            open(f"{self.output_dir}/output_info.json", "w"),
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(
        task=args.task,
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
    )
    model_adapter.run()
