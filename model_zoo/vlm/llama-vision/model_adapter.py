from typing import Dict, Any

import torch

import time
from transformers import MllamaForConditionalGeneration, AutoProcessor

from flagevalmm.server.utils import (
    parse_args,
    process_images_symbol,
    load_pil_image,
    default_collate_fn,
)
from flagevalmm.server.server_dataset import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)
        question_id = data["question_id"]
        qs, idx = process_images_symbol(data["question"])
        image_list, _ = load_pil_image(
            data["img_path"], idx, reduplicate=True, reqiures_img=True
        )
        return question_id, qs, image_list

    def __len__(self):
        return self.length


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]

        torch.set_grad_enabled(False)
        with self.accelerator.main_process_first():
            model = MllamaForConditionalGeneration.from_pretrained(
                ckpt_path,
                torch_dtype=torch.bfloat16,
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
                },
            )
        # add question
        messages[-1]["content"].append(
            {
                "type": "text",
                "text": query,
            },
        )
        return messages

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        cnt = 0

        data_loader = self.create_data_loader(
            CustomDataset,
            task_name,
            collate_fn=default_collate_fn,
            batch_size=1,
            num_workers=2,
        )
        for question_id, question, images in data_loader:
            if cnt == 1:
                start_time = time.perf_counter()
            cnt += 1
            messages = self.build_message(question[0], images[0])

            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            print(prompt)
            print(cnt)

            inputs = self.processor(
                images[0], prompt, add_special_tokens=False, return_tensors="pt"
            ).to(self.model.device)

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
        local_mode=args.local_mode,
        task_names=args.tasks,
        output_dir=args.output_dir,
        model_path=args.model,
        debug=args.debug,
        quiet=args.quiet,
    )
    model_adapter.run()
