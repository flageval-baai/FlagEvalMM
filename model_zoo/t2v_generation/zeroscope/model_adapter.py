import torch
import json
from diffusers import DiffusionPipeline
from tqdm import tqdm
from typing import Dict, Any
from flagevalmm.server.utils import get_data, parse_args
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from diffusers.utils import export_to_video


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict) -> None:
        self.tasks = task_info["task_names"]

        # pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
        pipe = DiffusionPipeline.from_pretrained(
            task_info["model_path"], torch_dtype=torch.float16
        )

        self.pipe = pipe.to("cuda")

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        text_num = meta_info["length"]
        output_dir = meta_info["output_dir"]
        output_info = []
        for i in tqdm(range(text_num)):
            response = get_data(i, task_name, self.server_ip, self.server_port)
            prompt, question_id = response["prompt"], response["id"]
            for j in range(1):
                video_frames = self.pipe(prompt, num_frames=24).frames[0]
                video_out_name = f"{output_dir}/{question_id}-{j}.mp4"
                export_to_video(video_frames, video_out_name)
            output_info.append(
                {"prompt": prompt, "id": question_id, "video_path": video_out_name}
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
