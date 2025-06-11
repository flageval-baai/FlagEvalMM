import torch
import json
from diffusers import FluxPipeline
from typing import Dict, Any
import os
from flagevalmm.server.utils import get_data, parse_args
from flagevalmm.models.base_model_adapter import BaseModelAdapter


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict) -> None:
        pipe = FluxPipeline.from_pretrained(
            task_info["model_path"], torch_dtype=torch.bfloat16, safety_checker=None
        )
        self.pipe = pipe.to("cuda").to(torch.float16)
        self.guidance_scale = task_info.get("guidance_scale", 3.5)
        self.num_inference_steps = task_info.get("num_inference_steps", 50)
        self.max_sequence_length = task_info.get("max_sequence_length", 512)

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        text_num = meta_info["length"]
        output_dir = meta_info["output_dir"]
        output_info = []
        for i in range(text_num):
            response = get_data(i, task_name, self.server_ip, self.server_port)
            prompt, question_id = response["prompt"], response["id"]
            image = self.pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                max_sequence_length=self.max_sequence_length,
                generator=torch.Generator("cpu").manual_seed(0),
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
