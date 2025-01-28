from typing import Dict, Any
from transformers import AutoProcessor

from flagevalmm.server.utils import (
    process_images_symbol,
    load_pil_image,
    parse_args,
    default_collate_fn,
)
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.server import ServerDataset

from vllm import LLM, SamplingParams


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)

        question_id = data["question_id"]
        qs = data["question"]

        qs, idx = process_images_symbol(qs)

        image_list, _ = load_pil_image(
            data["img_path"], idx, reduplicate=True, reqiures_img=True
        )
        return question_id, qs, image_list


class ModelAdapterVllm(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        with self.accelerator.main_process_first():
            self.model = LLM(
                model=ckpt_path,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 9},
                max_model_len=32768,
                mm_processor_kwargs={"max_dynamic_patch": 4},
            )

        self.processor = AutoProcessor.from_pretrained(
            ckpt_path, trust_remote_code=True
        )
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [self.processor.convert_tokens_to_ids(i) for i in stop_tokens]
        self.sampling_params = SamplingParams(
            max_tokens=1024, temperature=0.0, stop_token_ids=stop_token_ids
        )

    def build_prompt(self, question, images):
        placeholders = "\n".join(
            f"Image-{i}: <image>\n" for i, _ in enumerate(images, start=1)
        )
        conversation = [
            {"role": "user", "content": f"{placeholders}\n{question}"},
        ]
        prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        return prompt

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        data_loader = self.create_data_loader(
            CustomDataset,
            task_name,
            task_type=meta_info["type"],
            collate_fn=default_collate_fn,
            batch_size=1,
            num_workers=2,
        )
        for question_id, batch_question, batch_images in data_loader:
            inputs = []
            for question, images in zip(batch_question, batch_images):
                prompt = self.build_prompt(question, images)
                inputs.append({"prompt": prompt, "multi_modal_data": {"image": images}})
            outputs = self.model.generate(inputs, sampling_params=self.sampling_params)
            for i, output in enumerate(outputs):
                response = output.outputs[0].text.strip()
                print(f"{batch_question[i]}\n{response}\n\n")
                results.append(
                    {
                        "question_id": question_id[i],
                        "answer": response,
                        "prompt": batch_question[i],
                    }
                )
        self.save_result(results, meta_info)


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapterVllm(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        extra_cfg=args.cfg,
    )
    model_adapter.run()
