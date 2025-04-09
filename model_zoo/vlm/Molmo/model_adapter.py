from typing import Dict, Any
import torch

from transformers import AutoProcessor

from flagevalmm.server.utils import (
    parse_args,
    process_images_symbol,
    load_pil_image,
    default_collate_fn,
)
from flagevalmm.server.server_dataset import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.common.image_utils import concat_images
from vllm import LLM, SamplingParams


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)
        question_id = data["question_id"]
        img_path = data["img_path"]
        qs, idx = process_images_symbol(data["question"])
        image_list, idx = load_pil_image(
            img_path, idx, reqiures_img=True, reduplicate=True
        )
        image_list = concat_images(image_list) if len(image_list) > 1 else image_list[0]
        return question_id, qs, image_list

    def __len__(self):
        return self.length


class ModelAdapter(BaseModelAdapter):

    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        tensor_parallel_size = task_info.get("tensor_parallel_size", 1)
        torch.set_grad_enabled(False)
        self.model = LLM(
            model=ckpt_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.sampling_params = SamplingParams(
            max_tokens=1024, temperature=0.0, stop_token_ids=None
        )
        self.processor = AutoProcessor.from_pretrained(
            ckpt_path, trust_remote_code=True, torch_dtype="auto", device_map="auto"
        )

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        data_loader = self.create_data_loader(
            CustomDataset,
            task_name,
            collate_fn=default_collate_fn,
            batch_size=4,
            num_workers=2,
        )
        for question_id, batch_question, batch_images in data_loader:
            inputs = []
            for question, images in zip(batch_question, batch_images):
                inputs.append(
                    {"prompt": question, "multi_modal_data": {"image": images}}
                )

            outputs = self.model.generate(inputs, sampling_params=self.sampling_params)

            # remove input tokens
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
