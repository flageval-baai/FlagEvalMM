import torch
import re
from typing import Dict, Any
import time
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from flagevalmm.server import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.server.utils import parse_args, process_images_symbol
from qwen_vl_utils import process_vision_info


def parse_think_answer_string(text_string):
    think_content = None
    answer_content = None

    think_match = re.search(r"<think>(.*?)</think>", text_string, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()

    answer_match = re.search(r"<answer>(.*?)</answer>", text_string, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()  # .strip()

    return think_content, answer_content


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)
        question_id = data["question_id"]
        if self.task_type == "video_qa":
            img_path = data["video_path"]
        else:
            img_path = data["img_path"]
        qs = data["question"]
        qs, idx = process_images_symbol(qs)
        qs = qs.strip()
        idx = set(idx)
        img_path_idx = []
        for i in idx:
            if i < len(img_path):
                img_path_idx.append(img_path[i])
            else:
                print("[warning] image index out of range")
        return question_id, img_path_idx, qs


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        torch.set_grad_enabled(False)
        with self.accelerator.main_process_first():
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                ckpt_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )

        model = self.accelerator.prepare_model(model, evaluation_mode=True)
        self.tokenizer = tokenizer
        if hasattr(model, "module"):
            model = model.module
        self.model = model
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            ckpt_path, min_pixels=min_pixels, max_pixels=max_pixels
        )

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
                {"type": "image", "image": img_path},
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
            batch_size=1,
            num_workers=0,
            task_type=meta_info["type"],
        )
        for question_id, img_path, qs in data_loader:
            if cnt == 1:
                start_time = time.perf_counter()
            cnt += 1

            question_id = question_id[0]
            img_path_flaten = [p[0] for p in img_path]
            qs = qs[0]
            qs = f"{qs}<think></think><answer>"
            messages = self.build_message(qs, image_paths=img_path_flaten)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            content = response.strip()
            # Split reasoning and answer if present
            if "</think>" in content:
                reason, answer = parse_think_answer_string(content)
            else:
                answer = content
            self.accelerator.print(f"{qs}\n{response}\n\n")
            results.append({"question_id": question_id, "answer": answer, "prompt": qs})
        rank = self.accelerator.state.local_process_index

        self.save_result(results, meta_info, rank)
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
