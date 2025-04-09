import re
import time
from typing import Dict, Any, Callable, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

from flagevalmm.server.utils import parse_args, load_pil_image
from flagevalmm.server.server_dataset import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.common.video_utils import load_image_or_video


def replace_images_symbol(text):
    pattern = r"<image (\d+)>"
    matches = re.findall(pattern, text)
    for i, match in enumerate(matches):
        text = text.replace(f"<image {match}>", f"<|image_{i + 1}|>", 1)
    return text, [int(num) - 1 for num in matches]


class CustomDataset(ServerDataset):
    def __init__(self, task_name, server_ip, server_port, timeout, max_num_frames=8):
        super().__init__(task_name, server_ip, server_port, timeout)
        self.max_num_frames = max_num_frames

    def __getitem__(self, index):
        data = self.get_data(index)
        qs, idx = replace_images_symbol(data["question"])
        question_id = data["question_id"]
        if data.get("img_path"):
            img_path = data["img_path"]
            image_list, idx = load_pil_image(img_path, idx, reqiures_img=True)
        else:
            img_path = data["video_path"]
            image_list = load_image_or_video(
                data["video_path"],
                max_num_frames=self.max_num_frames,
                return_tensors=False,
            )
        # add dummy image if no image is provided
        if len(img_path) == 0:
            qs = f"<|image_1|>\n{qs}"
        return question_id, qs, image_list

    def __len__(self):
        return self.length


def collate_fn(batch):
    question_ids = [item[0] for item in batch]
    questions = [item[1] for item in batch]
    images_list = [item[2] for item in batch]

    return question_ids, questions, images_list


class ModelAdapter(BaseModelAdapter):

    def create_data_loader(
        self,
        dataset_cls: type[CustomDataset],
        task_name: str,
        collate_fn: Optional[Callable] = None,
        batch_size: int = 1,
        num_workers: int = 2,
        max_num_frames: int = 8,
    ):
        dataset = dataset_cls(
            task_name,
            self.server_ip,
            self.server_port,
            self.timeout,
            max_num_frames=max_num_frames,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )
        return data_loader

    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        self.max_num_frames = task_info.get("max_num_frames", 8)

        torch.set_grad_enabled(False)
        with self.accelerator.main_process_first():
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                device_map="cuda",
                trust_remote_code=True,
                torch_dtype="auto",
                _attn_implementation="flash_attention_2",
            )
            model = model.to(device="cuda", dtype=torch.bfloat16).eval()

            self.processor = AutoProcessor.from_pretrained(
                ckpt_path, trust_remote_code=True, num_crops=4
            )
        model = self.accelerator.prepare_model(model, evaluation_mode=True)
        if hasattr(model, "module"):
            model = model.module
        self.model = model

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        cnt = 0
        data_loader = self.create_data_loader(
            CustomDataset,
            task_name,
            collate_fn=collate_fn,
            batch_size=1,
            max_num_frames=self.max_num_frames,
        )

        for question_id, question, images in data_loader:
            if cnt == 1:
                start_time = time.perf_counter()
            cnt += 1
            image_tokens = (
                "".join("<image_start><image><image_end>\n" for _ in images[0])
                + "<image_end>\n"
            )
            prompt_content = [
                {"role": "user", "content": f"{image_tokens}{question[0]}"}
            ]

            prompt = self.processor.tokenizer.apply_chat_template(
                prompt_content, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                texts=prompt, images=images[0], return_tensors="pt"
            ).to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
            inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)
            inputs = inputs.to("cuda").to(torch.bfloat16)

            generation_args = {
                "max_new_tokens": 1024,
                "temperature": 0.0,
                "do_sample": False,
            }

            generate_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **generation_args,
            )

            # remove input tokens
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            self.accelerator.print(f"{question[0]}\n{response}\n\n")
            results.append(
                {
                    "question_id": question_id[0],
                    "answer": response.strip(),
                    "prompt": question[0],
                }
            )
        rank = self.accelerator.state.local_process_index

        # save results for the rank
        self.save_result(results, meta_info, rank=rank)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            correct_num = self.collect_results_and_save(meta_info)
            total_time = time.perf_counter() - start_time
            print(
                f"Total time: {total_time}\nAverage time:{total_time / cnt}\nResults_collect number: {correct_num}"
            )

        print("rank", rank, "finished")


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
