from typing import Dict, Any

from flagevalmm.models.base_model_adapter import BaseModelAdapter
from modeling.utils import (
    load_model_and_tokenizer,
    load_gen_model_and_tokenizer,
    build_transform,
    process_conversation,
)
import torch
from flagevalmm.common.logger import get_logger
from flagevalmm.server import ServerDataset
from flagevalmm.server.utils import (
    process_images_symbol,
    load_pil_image,
    default_collate_fn,
    parse_args,
    RunCfg,
)
from modeling.bagel.qwen2_navit import NaiveCache
from PIL import Image
import os
import json
import time

logger = get_logger(__name__)


class BagelDataset(ServerDataset):
    """Load images as PIL and keep question text aligned with placeholders."""

    def __getitem__(self, index):
        data = self.get_data(index)
        question_id = data["question_id"]
        question = data["question"]
        img_paths = data["img_path"]
        if isinstance(img_paths, str):
            img_paths = [img_paths]

        # Extract referenced image indices (if any) to keep order consistent
        question, img_idx = process_images_symbol(question)
        images, _ = load_pil_image(
            img_paths, img_idx, reqiures_img=True, reduplicate=False
        )

        return question_id, question, images


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict) -> None:
        ckpt_path = task_info["model_path"]
        model_cfg = task_info.get("model_cfg", {}) or {}
        # Prefer runtime config `model.extra_args`, fallback to legacy top-level `extra_args`.
        extra_args = model_cfg.get("extra_args", None)
        if not isinstance(extra_args, dict):
            extra_args = task_info.get("extra_args", {}) or {}
        self.extra_args: Dict[str, Any] = extra_args
        use_gen_model = bool(extra_args.get("use_gen_model", True))
        max_latent_size = int(extra_args.get("max_latent_size", 64))

        torch.set_grad_enabled(False)
        with self.accelerator.main_process_first():
            print(f"use_gen_model: {use_gen_model}")
            if use_gen_model:
                model, self.tokenizer, self.new_token_ids, vae_model = (
                    load_gen_model_and_tokenizer(
                        ckpt_path, max_latent_size=max_latent_size
                    )
                )
                print('Successfully loaded generation model')
                # Keep generation components available for T2I.
                self.vae_model = vae_model.to(torch.bfloat16).cuda().eval()
                self.gen_model = model
                self._t2i_initialized = True
            else:
                model, self.tokenizer, self.new_token_ids = load_model_and_tokenizer(
                    ckpt_path
                )
            model = model.to(torch.bfloat16).cuda().eval()
        model = self.accelerator.prepare_model(model, evaluation_mode=True)
        if hasattr(model, "module"):
            model = model.module
        self.model = model
        self.image_transform = build_transform()


    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        # Determine task type from task name
        is_t2i = "t2i" in meta_info["type"].lower()
        logger.info(
            f"Running {task_name, meta_info} as {'T2I' if is_t2i else 'VQA'} task"
        )
        self.dataset = ServerDataset(
            task_name,
            task_type=meta_info["type"],
            task_manager=self.task_manager,
        )
        if is_t2i:
            self._run_t2i_task(task_name, meta_info)
        else:
            self._run_vqa_task(task_name, meta_info)

    def _run_t2i_task(self, task_name: str, meta_info: Dict[str, Any]):
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return

        # T2I components are expected to be initialized in `model_init`
        # when `model.use_gen_model=true`.
        if not hasattr(self, "gen_model") or not hasattr(self, "vae_model"):
            raise RuntimeError(
                "T2I requires generation components, but they are not initialized. "
                "Please set `model.use_gen_model=true` in runtime config."
            )
        text_num = meta_info["length"]
        output_dir = meta_info["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # T2I sampling knobs: prefer runtime `extra_args`, fallback to dataset meta_info defaults.
        extra_args = getattr(self, "extra_args", {}) or {}
        save_items = bool(extra_args.get("save_items", True))
        num_images = int(extra_args.get("num_images", 4))
        batch_size = int(extra_args.get("batch_size", 1))
        cfg_scale = float(extra_args.get("cfg_scale", 4.0))
        resolution = int(extra_args.get("resolution", 1024))
        num_timesteps = int(extra_args.get("num_timesteps", 50))
        cfg_interval = extra_args.get("cfg_interval", [0, 1.0])
        cfg_renorm_min = float(extra_args.get("cfg_renorm_min", 0.0))
        timestep_shift = float(extra_args.get("timestep_shift", 3.0))

        output_info: list[dict[str, Any]] = []

        for idx in range(text_num):
            data = self.task_manager.get_data(task_name, idx)
            prompt = data.get("prompt") or data.get("question")
            question_id = str(data.get("id") or data.get("question_id") or idx)

            image_list = self._generate_images(
                prompt=prompt,
                num_images=num_images,
                batch_size=batch_size,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                resolution=resolution,
            )

            sample_dir = os.path.join(output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)

            image_names: list[str] = []
            for i, sample in enumerate(image_list):
                image_name = f"{question_id}_{i:05}.png"
                sample_path = os.path.join(sample_dir, image_name)
                sample.save(sample_path)
                image_names.append(image_name)

            output_info.append(
                {
                    "question_id": question_id,
                    "id": question_id,
                    "prompt": prompt,
                    "images": image_names,
                }
            )
            if save_items:
                self.save_item(
                    {
                        "question_id": question_id,
                        "id": question_id,
                        "prompt": prompt,
                        "images": image_names,
                    },
                    question_id=question_id,
                    meta_info=meta_info,
                )

        # save results (main process only)
        self.save_result(output_info, meta_info, rank=None)
        self.accelerator.wait_for_everyone()

    def _run_vqa_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        cnt = 0
        start_time = None
        extra_args = getattr(self, "extra_args", {}) or {}
        num_workers = int(extra_args.get("num_workers", 2))
        max_new_tokens = int(extra_args.get("max_new_tokens", 1024))
        save_items = bool(extra_args.get("save_items", True))
        data_loader = self.create_data_loader(
            BagelDataset,
            task_name,
            collate_fn=default_collate_fn,
            batch_size=1,
            num_workers=num_workers,
        )

        for question_id, question, images in data_loader:
            if cnt == 0:
                start_time = time.perf_counter()
            cnt += 1

            # unwrap batch dimension (batch=1)
            images, prompt = process_conversation(images[0], question[0])

            pred = self.model.chat(
                self.tokenizer,
                self.new_token_ids,
                self.image_transform,
                images=images,
                prompt=prompt,
                max_length=max_new_tokens,
            )

            results.append(
                {
                    "answer": pred,
                    "question_id": question_id[0],
                    "prompt": prompt,
                }
            )
            if save_items:
                self.save_item(
                    {
                        "answer": pred,
                        "question_id": question_id[0],
                        "prompt": prompt,
                    },
                    question_id=str(question_id[0]),
                    meta_info=meta_info,
                )

        rank = self.accelerator.state.local_process_index

        # save results for the rank
        self.save_result(results, meta_info, rank=rank)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            correct_num = self.collect_results_and_save(meta_info)
            if start_time is not None and cnt > 0:
                total_time = time.perf_counter() - start_time
                print(
                    f"Total time: {total_time}\nAverage time:{total_time / cnt}\nResults_collect number: {correct_num}"
                )

        print("rank", rank, "finished")

    def _move_generation_input_to_device(self, generation_input: Dict[str, Any], device):
        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(device)
        return generation_input

    @torch.inference_mode()
    def _generate_images(
        self,
        prompt: str,
        num_images: int,
        batch_size: int,
        cfg_scale: float,
        cfg_interval,
        cfg_renorm_min: float,
        timestep_shift: float,
        num_timesteps: int,
        resolution: int,
    ):
        device = next(self.gen_model.parameters()).device
        image_list = []

        assert batch_size == 1, "batch_size must be 1 for T2I"
        for _ in range(0, num_images, batch_size):
            curr_batch = min(batch_size, num_images - len(image_list))

            past_key_values = NaiveCache(self.gen_model.config.llm_config.num_hidden_layers)
            newlens = [0] * curr_batch
            new_rope = [0] * curr_batch

            generation_input, newlens, new_rope = self.model.prepare_prompts(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                prompts=[prompt] * curr_batch,
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
            )
            generation_input = self._move_generation_input_to_device(generation_input, device)

            with torch.no_grad():
                # Keep autocast dtype consistent with model weights (bf16 by default).
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)

            generation_input = self.gen_model.prepare_vae_latent(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                image_sizes=[(resolution, resolution)] * curr_batch,
                new_token_ids=self.new_token_ids,
            )
            generation_input = self._move_generation_input_to_device(generation_input, device)

            cfg_past_key_values = NaiveCache(self.gen_model.config.llm_config.num_hidden_layers)
            cfg_newlens = [0] * curr_batch
            cfg_new_rope = [0] * curr_batch

            generation_input_cfg = self.gen_model.prepare_vae_latent_cfg(
                curr_kvlens=cfg_newlens,
                curr_rope=cfg_new_rope,
                image_sizes=[(resolution, resolution)] * curr_batch,
            )
            generation_input_cfg = self._move_generation_input_to_device(generation_input_cfg, device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    unpacked_latent = self.model.generate_image(
                        past_key_values=past_key_values,
                        num_timesteps=num_timesteps,
                        cfg_text_scale=cfg_scale,
                        cfg_interval=cfg_interval,
                        cfg_renorm_min=cfg_renorm_min,
                        timestep_shift=timestep_shift,
                        cfg_text_past_key_values=cfg_past_key_values,
                        cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                        cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                        cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                        cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                        **generation_input,
                    )

            for latent in unpacked_latent:
                latent = latent.reshape(1, resolution // 16, resolution // 16, 2, 2, 16)
                latent = torch.einsum("nhwpqc->nchpwq", latent)
                latent = latent.reshape(1, 16, resolution // 8, resolution // 8)
                image = self.vae_model.decode(latent.to(device))
                tmpimage = (
                    (image * 0.5 + 0.5)
                    .clamp(0, 1)[0]
                    .permute(1, 2, 0)
                    .mul(255)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                tmpimage = Image.fromarray(tmpimage)
                image_list.append(tmpimage)

        return image_list

if __name__ == "__main__":
    args = parse_args()
    defaults_obj = RunCfg()
    model_adapter = ModelAdapter(
        server_ip=defaults_obj.server.ip,
        server_port=defaults_obj.server.port,
        timeout=defaults_obj.server.timeout,
        extra_cfg=args.cfg,
        local_mode=defaults_obj.server.local_mode,
        task_names=None,
    )
    model_adapter.run()
