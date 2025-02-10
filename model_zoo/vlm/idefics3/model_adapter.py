from typing import Dict, Any
import torch

import time
from transformers import AutoProcessor, AutoModelForVision2Seq

from flagevalmm.server.utils import parse_args, process_images_symbol, load_pil_image
from flagevalmm.server.server_dataset import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
import re


def get_images_symbol_index(text):
    pattern = r"<image (\d+)>"
    matches = re.findall(pattern, text)
    return [int(num) - 1 for num in matches]


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)
        qs, idx = process_images_symbol(data["question"], "")
        question_id = data["question_id"]
        img_path = data["img_path"]
        image_list, idx = load_pil_image(
            img_path, idx, reqiures_img=True, reduplicate=True
        )

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
            model = (
                AutoModelForVision2Seq.from_pretrained(
                    ckpt_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                .to(device="cuda")
                .eval()
            )

            self.processor = AutoProcessor.from_pretrained(ckpt_path)
        model = self.accelerator.prepare_model(model, evaluation_mode=True)
        if hasattr(model, "module"):
            model = model.module
        self.model = model

    def build_message(
        self,
        query: str,
        image_paths=[],
    ) -> str:
        messages = []
        messages.append(
            {
                "role": "user",
                "content": [],
            },
        )
        for img_path in image_paths:
            messages[-1]["content"].append(
                {
                    "type": "image",
                }
            )
        # add question
        messages[-1]["content"].append(
            {
                "type": "text",
                "text": query,
            }
        )
        return messages

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        cnt = 0

        data_loader = self.create_data_loader(
            CustomDataset, task_name, collate_fn=collate_fn, batch_size=1, num_workers=2
        )
        for question_id, question, images in data_loader:
            if cnt == 1:
                start_time = time.perf_counter()
            cnt += 1

            messages = self.build_message(question[0], images[0])

            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            print(cnt)
            inputs = self.processor(
                images=images[0],
                text=prompt,
                # add_special_tokens=False,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            output = self.model.generate(**inputs, max_new_tokens=1024)
            generated_tokens = output[0, inputs["input_ids"].size(1) :]
            response = self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
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
