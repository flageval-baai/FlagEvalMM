import torch

import time
from PIL import Image
import re
import math

from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T

from torchvision.transforms.functional import InterpolationMode

from flagevalmm.server import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.server.utils import parse_args


# modified from https://huggingface.co/OpenGVLab/InternVL2_5-78B
def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2_5-1B": 24,
        "InternVL2_5-2B": 24,
        "InternVL2_5-4B": 36,
        "InternVL2_5-8B": 32,
        "InternVL2_5-26B": 48,
        "InternVL2_5-38B": 64,
        "InternVL2_5-78B": 80,
    }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    device_map["language_model.model.rotary_emb"] = 0

    return device_map


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def replace_images_symbol(text):
    pattern = r"<image (\d+)>"

    matches = [int(num) - 1 for num in re.findall(pattern, text)]

    new_text = re.sub("<image ", "<Image-", text)
    return new_text, matches


class CustomDataset(ServerDataset):
    def __getitem__(self, index):
        data = self.get_data(index)

        question_id = data["question_id"]
        img_paths = data["img_path"]
        qs, idx = replace_images_symbol(data["question"])

        pixel_values = []
        num_patches_list = []
        for img_path in img_paths:
            pixel_values.append(load_image(img_path, max_num=12).to(torch.bfloat16))
            num_patches_list.append(pixel_values[-1].size(0))

        image_list_idx = []
        image_size_idx = []
        idx = set(idx)
        for i in idx:
            if i < len(pixel_values):
                image_list_idx.append(pixel_values[i])
            else:
                print("[warning] image index out of range")
                image_list_idx.append(pixel_values[-1])
            image_size_idx.append(image_list_idx[-1].size(0))
        pixel_values = image_list_idx
        num_patches_list = image_size_idx
        if len(pixel_values) == 0:
            image_tensor = None
            num_patches_list = None
        else:
            image_tensor = torch.cat(pixel_values, dim=0)
            if len(idx) > 1:
                qs = "\n".join([f"Image-{i + 1}: <image>" for i in idx]) + "\n" + qs
            else:
                qs = "<image>\n" + qs.replace("<Image-1>", "")
        return qs, image_tensor, question_id, num_patches_list

    def __len__(self):
        return self.length


def collate_fn(batch):
    question_ids = [item[2] for item in batch]
    questions = [item[0] for item in batch]
    images_list = [item[1] for item in batch]
    num_patches_list = [item[3] for item in batch]

    return questions, images_list, question_ids, num_patches_list


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info):
        ckpt_path = task_info["model_path"]
        torch.set_grad_enabled(False)
        with self.accelerator.main_process_first():
            tokenizer = AutoTokenizer.from_pretrained(
                ckpt_path, trust_remote_code=True, use_fast=False
            )
            device_map = split_model(ckpt_path.split("/")[-1])
            model = AutoModel.from_pretrained(
                ckpt_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=device_map,
                attn_implementation="flash_attention_2",
            ).eval()

        model = self.accelerator.prepare_model(model, evaluation_mode=True)
        self.tokenizer = tokenizer
        if hasattr(model, "module"):
            model = model.module
        self.model = model
        self.generation_config = dict(
            do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1
        )

    def run_one_task(self, task_name, meta_info):
        results = []
        cnt = 0
        data_loader = self.create_data_loader(
            CustomDataset, task_name, collate_fn, batch_size=1, num_workers=0
        )
        for query, image_tensor, question_id, num_patches_list in data_loader:
            if cnt == 1:
                start_time = time.perf_counter()
            cnt += 1
            image_tensor = image_tensor[0]
            if image_tensor is not None:
                image_tensor = image_tensor.cuda()
            response = self.model.chat(
                self.tokenizer,
                image_tensor,
                query[0],
                self.generation_config,
                num_patches_list=num_patches_list[0],
            )
            self.accelerator.print(f"{query[0]}\n{response}\n\n")
            results.append(
                {
                    "question_id": question_id[0],
                    "answer": response.strip(),
                    "prompt": query[0],
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
