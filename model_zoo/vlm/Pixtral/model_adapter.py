from typing import Optional, List, Dict, Any

import time

from flagevalmm.server.utils import parse_args
from flagevalmm.server.server_dataset import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.prompt.prompt_tools import encode_image

import re
from vllm import LLM
from vllm import SamplingParams


def get_images_symbol_index(text):
    pattern = r"<image (\d+)>"
    matches = re.findall(pattern, text)
    return [int(num) - 1 for num in matches]


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)
        question_id = data["question_id"]
        img_path = data["img_path"]
        qs = data["question"]
        idx = get_images_symbol_index(qs)
        img_path_idx = []
        idx = set(idx)
        for i in idx:
            if i < len(img_path):
                img_path_idx.append(img_path[i])
            else:
                print("[warning] image index out of range")
        return question_id, img_path_idx, qs


class ModelAdapterVllm(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        with self.accelerator.main_process_first():
            self.model = LLM(
                model=ckpt_path,
                tokenizer_mode="mistral",
                limit_mm_per_prompt={"image": 18},
                max_model_len=32768,
            )
        self.sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)

    def build_message(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        image_paths: List[str] = [],
        past_messages: Optional[List] = None,
    ) -> str:
        messages = past_messages if past_messages else []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    },
                ],
            },
        )
        for img_path in image_paths:
            base64_image = encode_image(img_path, max_size=4 * 1024 * 1024)
            messages[-1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            )
        return messages

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        cnt = 0
        data_loader = self.create_data_loader(
            CustomDataset, task_name, batch_size=1, num_workers=0
        )
        start_time = time.perf_counter()
        for question_id, img_path, qs in data_loader:
            cnt += 1
            question_id = question_id[0]
            img_path_flaten = [p[0] for p in img_path]
            qs = qs[0]

            messages = self.build_message(qs, image_paths=img_path_flaten)
            try:
                outputs = self.model.chat(
                    messages=messages, sampling_params=self.sampling_params
                )
                result = outputs[0].outputs[0].text
            except Exception as e:
                print(e)
                result = str(e)
                engine = self.model.llm_engine
                # get all request_id
                request_ids = []
                for scheduler in engine.scheduler:
                    for seq_group in scheduler.running:
                        request_ids.append(seq_group.request_id)
                engine.abort_request(request_ids)

            results.append(
                {"question_id": question_id, "question": qs, "answer": result}
            )
            self.accelerator.print(f"{qs}\n{result}\n\n")
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
    model_adapter = ModelAdapterVllm(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        extra_cfg={"model_path": "mistralai/Pixtral-12B-2409"},
    )
    model_adapter.run()
