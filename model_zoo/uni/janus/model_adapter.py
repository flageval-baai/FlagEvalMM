import torch
import json
import time
from typing import Dict, Any
import os
from flagevalmm.server.utils import (
    get_data,
    parse_args,
    default_collate_fn,
    process_images_symbol,
    load_pil_image,
)
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.server import ServerDataset
from flagevalmm.common.logger import get_logger

import PIL.Image
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

logger = get_logger(__name__)


@torch.inference_mode()
def generate_image(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.int
    ).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,  # noqa
        )
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    return visual_img


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)
        qs, idx = process_images_symbol(
            data["question"], dst_pattern="<image_placeholder>"
        )
        question_id = data["question_id"]
        img_path = data["img_path"]
        image_list, idx = load_pil_image(
            img_path, idx, reqiures_img=True, reduplicate=False
        )

        return question_id, qs, image_list


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict) -> None:
        ckpt_path = task_info["model_path"]

        torch.set_grad_enabled(False)
        with self.accelerator.main_process_first():
            self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
                ckpt_path
            )
            self.tokenizer = self.vl_chat_processor.tokenizer

            vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                ckpt_path, trust_remote_code=True
            )
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

        model = self.accelerator.prepare_model(vl_gpt, evaluation_mode=True)
        if hasattr(model, "module"):
            model = model.module
        self.model = model

    def build_message(
        self,
        query: str,
        image_paths=[],
    ) -> str:
        messages = [
            {
                "role": "User",
                "content": query,
                "images": image_paths,
            },
            {"role": "Assistant", "content": ""},
        ]
        return messages

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        # Determine task type from task name
        is_t2i = "t2i" in meta_info['type'].lower()
        logger.info(
            f"Running {task_name, meta_info} as {'T2I' if is_t2i else 'VQA'} task"
        )
        if is_t2i:
            self._run_t2i_task(task_name, meta_info)
        else:
            self._run_vqa_task(task_name, meta_info)

    def _run_t2i_task(self, task_name: str, meta_info: Dict[str, Any]):
        text_num = meta_info["length"]
        output_dir = meta_info["output_dir"]
        output_info = []

        for i in range(text_num):
            response = get_data(i, task_name, self.server_ip, self.server_port)
            prompt, question_id = response["prompt"], response["id"]
            conversation = [
                {
                    "role": "User",
                    "content": prompt,
                },
                {"role": "Assistant", "content": ""},
            ]

            sft_format = (
                self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=conversation,
                    sft_format=self.vl_chat_processor.sft_format,
                    system_prompt="",
                )
            )
            prompt = sft_format + self.vl_chat_processor.image_start_tag
            image = generate_image(
                self.model, self.vl_chat_processor, prompt, parallel_size=1
            )[0]
            image = PIL.Image.fromarray(image)
            image_out_name = f"{question_id}.png"
            image.save(os.path.join(output_dir, image_out_name))
            output_info.append(
                {"prompt": prompt, "id": question_id, "image": image_out_name}
            )

        json.dump(
            output_info,
            open(f"{output_dir}/{task_name}.json", "w"),
            indent=2,
            ensure_ascii=False,
        )

    def _run_vqa_task(self, task_name: str, meta_info: Dict[str, Any]):
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

            pil_images = images[0]
            prepare_inputs = self.vl_chat_processor(
                conversations=messages, images=pil_images, force_batchify=True
            ).to(self.model.device)

            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True,
            )

            response = self.tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
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
