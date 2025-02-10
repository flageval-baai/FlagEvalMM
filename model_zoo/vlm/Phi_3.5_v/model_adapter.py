from typing import Dict, Any
import torch

import time
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

from flagevalmm.server.utils import parse_args, load_pil_image
from flagevalmm.server.server_dataset import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
import re


def replace_images_symbol(text):
    pattern = r"<image (\d+)>"
    matches = re.findall(pattern, text)
    for i, match in enumerate(matches):
        text = text.replace(f"<image {match}>", f"<|image_{i + 1}|>", 1)
    return text, [int(num) - 1 for num in matches]


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)
        qs, idx = replace_images_symbol(data["question"])
        question_id = data["question_id"]
        img_path = data["img_path"]
        image_list, idx = load_pil_image(img_path, idx, reqiures_img=True)
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
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]

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
            CustomDataset, task_name, collate_fn=collate_fn, batch_size=1
        )

        for question_id, question, images in data_loader:
            if cnt == 1:
                start_time = time.perf_counter()
            cnt += 1
            messages = [{"role": "user", "content": question[0]}]

            prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print(cnt)

            inputs = self.processor(prompt, images[0], return_tensors="pt").to("cuda")

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
    )
    model_adapter.run()
