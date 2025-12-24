import json
import os
from typing import Any, Dict, List
import math
import torch
from diffusers import DiffusionPipeline
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.server.utils import parse_args
logger = get_logger(__name__)

class ModelAdapter(BaseModelAdapter):
    """Minimal T2I adapter for the Qwen-Image diffusion pipeline."""

    def model_init(self, task_info: Dict) -> None:
        # Decide device/dtype
        
        self.device = (
            self.accelerator.device
            if getattr(self, "accelerator", None) is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Load pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            task_info["model_path"], torch_dtype=torch.bfloat16, device_map="cuda"
        ).to(self.device)

        # Runtime knobs (all optional; fall back to sane defaults)
        model_cfg = task_info.get("model_cfg", {}) or {}
        extra_args = model_cfg.get("extra_args", {}) or {}
        self.cfg_scale = float(extra_args.get("cfg_scale", 4.0))
        self.save_items = bool(extra_args.get("save_items", True))
        resolution = int(extra_args.get("resolution", 1024))
        self.width = math.sqrt(resolution)
        self.height = math.sqrt(resolution)
        self.num_timesteps = int(extra_args.get("num_timesteps", 50))
        self.seed = int(extra_args.get("seed", 0))
        # Deterministic seed on CPU by default; users can change if needed.
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

    def _run_single_prompt(self, prompt: str, question_id: str, output_dir: str) -> List[str]:
        """Generate images for one prompt and return saved filenames."""
        results = self.pipe(
            prompt=prompt,
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_timesteps,
            true_cfg_scale=self.cfg_scale,
            generator=self.generator,
        ).images

        image_names: List[str] = []
        os.makedirs(output_dir, exist_ok=True)
        for idx, image in enumerate(results):
            name = f"{question_id}_{idx:05}.png" if self.num_images > 1 else f"{question_id}.png"
            image.save(os.path.join(output_dir, name))
            image_names.append(name)
        return image_names

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        task_type = meta_info["type"].lower()
        is_t2i = "t2i" in task_type
        is_i2i = "i2i" in task_type
        is_vqa = "vqa" in task_type
        logger.info(
            f"Running {task_name, meta_info} as "
            f"{'T2I' if is_t2i else ('I2I' if is_i2i else 'VQA')} task"
        )
        if is_i2i:
            self._run_i2i_task(task_name, meta_info)
        elif is_t2i:
            self._run_t2i_task(task_name, meta_info)
        elif is_vqa:
            self._run_vqa_task(task_name, meta_info)
        else:
            raise NotImplementedError(f"Qwen-Image adapter currently supports T2I, I2I and VQA, but got {task_type}")

    def _run_t2i_task(self, task_name: str, meta_info: Dict[str, Any]):
        data_len = meta_info["length"]
        output_dir = meta_info["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.get_items_dir(meta_info), exist_ok=True)
        output_info = []
        world_size = (
            self.accelerator.state.num_processes if self.accelerator is not None else 1
        )
        rank = (
            self.accelerator.state.local_process_index
            if self.accelerator is not None
            else 0
        )
        for idx in range(rank, data_len, world_size):
            data = self.task_manager.get_data(task_name, idx)
            print(f"data: {data}")
            prompt = data.get("prompt") or data.get("question")
            question_id = str(data.get("id") or data.get("question_id") or idx)
            image_names = self._run_single_prompt(prompt, question_id, output_dir)
            output_info.append(
                {
                 "question_id": question_id,
                 "id": question_id,
                 "prompt": prompt,
                 "images": image_names
                 }
            )

        with open(os.path.join(output_dir, f"{task_name}.json"), "w") as f:
            json.dump(output_info, f, indent=2, ensure_ascii=False)
            
    def _run_i2i_task(self, task_name: str, meta_info: Dict[str, Any]):
        pass
    
    def _run_vqa_task(self, task_name: str, meta_info: Dict[str, Any]):
        pass

if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(
        extra_cfg=args.cfg,
        task_names=None,
    )
    model_adapter.run()

