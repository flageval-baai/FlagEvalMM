import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
from typing import Dict, Any
from flagevalmm.server.utils import parse_args, get_retrieval_data
from flagevalmm.models.base_model_adapter import BaseModelAdapter
import numpy as np
import os


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict) -> None:
        self.tasks = task_info["task_names"]

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=task_info["model_path"],
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            task_info["model_path"], trust_remote_code=True
        )

    def get_video(self, video_id, task_name):
        response = get_retrieval_data(
            index=video_id,
            task_name=task_name,
            data_type="video",
            server_ip=self.server_ip,
            server_port=self.server_port,
            timeout=self.timeout,
        )
        return response["video_path"]

    def get_caption(self, caption_id, task_name):
        response = get_retrieval_data(
            index=caption_id,
            task_name=task_name,
            data_type="text",
            server_ip=self.server_ip,
            server_port=self.server_port,
            timeout=self.timeout,
        )
        return response["caption"]

    def process_single(self, text, video_path):
        num = len(text.split("\n"))
        question = f"You will be given {num} sentences and one video. Your task is to analyze the semantic and contextual similarity between each sentence and the video content, \
            then provide a absolute similarity score for each sentence on a scale from 0 to 1 (where 0 = no similarity, 1 = perfect match). Pleas return a list of {num} comma-separated scores in parentheses, \
            rounded to 2 decimal places (e.g., 0.91, 0.80, 0.55, 0.91, 0.72, 0.93, 0.88, 0.97, 0.86, 0.85). \
            Maintain strict consistency in scoring across all evaluations. \
            Scores should enable accurate video retrieval comparisons. The sentences are {text}"

        # Video conversation
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": 1,
                            "max_frames": 128,
                        },
                    },
                    {"type": "text", "text": question},
                ],
            },
        ]

        inputs = self.processor(conversation=conversation, return_tensors="pt")
        inputs = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = self.model.generate(**inputs, max_new_tokens=128)
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return response

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        text_num = meta_info["caption_number"]
        output_dir = meta_info["output_dir"]
        output = np.zeros((text_num, text_num), dtype=float)
        caption = [
            self.get_caption(caption_id, task_name) for caption_id in range(text_num)
        ]
        videos = [self.get_video(video_id, task_name) for video_id in range(text_num)]
        itd = 10

        for i in tqdm(range(text_num)):
            for j in range(0, text_num, itd):
                _s, _e = j, min(j + itd, text_num)
                text = "\n".join(caption[_s:_e])

                result = self.process_single(text, videos[i])
                arr = np.fromstring(result, dtype=float, sep=",")[: _e - _s]
                arr = np.pad(
                    arr, _e - _s - arr.shape[0], "constant", constant_values=(0.0)
                )

                cnt = 1
                while arr.shape[0] != _e - _s:
                    print(f"error ouput, repeat question. output:{result}")
                    result = self.process_single(text, videos[i])
                    arr = np.fromstring(result, dtype=float, sep=",")[: _e - _s]
                    arr = np.pad(
                        arr, _e - _s - arr.shape[0], "constant", constant_values=(0.0)
                    )
                    if cnt > 5:
                        arr = np.zeros(_e - _s, dtype=float)
                        break
                output[i][_s:_e] = arr

        # json.dump(
        #     output_info,
        #     open(f"{output_dir}/{task_name}.json", "w"),
        #     indent=2,
        #     ensure_ascii=False,
        # )
        full_save_path = os.path.join(output_dir, meta_info["name"])
        np.save("{}".format(full_save_path), output)


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
    # print(model_adapter.process_single(text,video_path))
