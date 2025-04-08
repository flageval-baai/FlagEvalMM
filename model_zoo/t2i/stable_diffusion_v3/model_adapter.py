import torch
import json
from diffusers import StableDiffusion3Pipeline
import os
from typing import Dict, Any
from flagevalmm.server.utils import get_data, parse_args
from flagevalmm.models.base_model_adapter import BaseModelAdapter


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict) -> None:
        self.tasks = task_info["task_names"]

        pipe = StableDiffusion3Pipeline.from_pretrained(
            task_info["model_path"], safety_checker=None, torch_dtype=torch.float16
        )
        self.pipe = pipe.to("cuda")

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        text_num = meta_info["length"]
        output_dir = meta_info["output_dir"]
        output_info = []
        for i in range(text_num):
            response = get_data(i, task_name, self.server_ip, self.server_port)
            prompt, question_id = response["prompt"], response["id"]
            image = self.pipe(
                prompt, num_inference_steps=28, guidance_scale=3.5
            ).images[0]
            image_out_name = f"{question_id}.png"
            image.save(os.path.join(output_dir, image_out_name))
            output_info.append(
                {"prompt": prompt, "id": question_id, "image": image_out_name}
            )

        json.dump(
            output_info,
            open(f"{output_dir}/{task_name}.json", "w"),
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    args = parse_args()

    model_adapter = ModelAdapter(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        extra_cfg=args.cfg,
        local_mode=args.local_mode,
        task_names=args.tasks,
        output_dir=args.output_dir,
        model_path=args.model,
        debug=args.debug,
        quiet=args.quiet,
    )
    model_adapter.run()
