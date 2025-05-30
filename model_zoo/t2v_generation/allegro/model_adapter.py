import torch
import json
from diffusers import AutoencoderKLAllegro, AllegroPipeline
from diffusers.utils import export_to_video
from typing import Dict, Any
import os
from flagevalmm.server.utils import get_data, parse_args
from flagevalmm.models.base_model_adapter import BaseModelAdapter


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict) -> None:
        self.skip_exists = task_info.get("skip_exists", False)
        vae = AutoencoderKLAllegro.from_pretrained(
            task_info["model_path"], subfolder="vae", torch_dtype=torch.float32
        )
        pipe = AllegroPipeline.from_pretrained(
            task_info["model_path"], vae=vae, torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        pipe.vae.enable_tiling()
        self.pipe = pipe
        self.positive_prompt = """
        (masterpiece), (best quality), (ultra-detailed), (unwatermarked),
        {}
        emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo,
        sharp focus, high budget, cinemascope, moody, epic, gorgeous
        """

        self.negative_prompt = """
        nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality,
        low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
        """

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        text_num = meta_info["length"]
        output_dir = meta_info["output_dir"]
        output_info = []
        for i in range(text_num):
            response = get_data(i, task_name, self.server_ip, self.server_port)
            prompt, question_id = response["prompt"], response["id"]
            video_out_name = f"{question_id}.mp4"
            if self.skip_exists and os.path.exists(
                os.path.join(output_dir, video_out_name)
            ):
                print(f"Skipping: {prompt}, id: {question_id}")
                continue
            prompt = prompt.format(prompt.lower().strip())
            video = self.pipe(
                prompt,
                negative_prompt=self.negative_prompt,
                guidance_scale=7.5,
                max_sequence_length=512,
                num_inference_steps=100,
                generator=torch.Generator(device="cuda:0").manual_seed(42),
            ).frames[0]

            export_to_video(video, os.path.join(output_dir, video_out_name), fps=24)
            output_info.append(
                {"prompt": prompt, "id": question_id, "video": video_out_name}
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
    )
    model_adapter.run()
