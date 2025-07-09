import json
import os
import os.path as osp
from typing import List

from datasets import load_dataset


def save_image(question_id, images, output_dir) -> List[str]:
    img_path_list = []
    for i, image in enumerate(images):
        image_path = osp.join("img", f"{question_id}_{i + 1}.jpg")
        img_path_list.append(image_path)
        full_image_path = osp.join(output_dir, image_path)
        os.makedirs(osp.dirname(full_image_path), exist_ok=True)
        try:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.save(full_image_path)
        except Exception as e:
            print(f"Error saving image {question_id}: {e}")
    return img_path_list


def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    # build output path
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    img_dir = osp.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    # load dataset
    data = load_dataset(data_dir, name=name, split=split)

    content = []
    for annotation in data:
        question_id = annotation["id"]
        question = annotation["question"]
        item = {
            "question_id": question_id,
            "sub_task": annotation["question_type"],
            "answer": annotation["answer"],
            "question_type": "multiple-choice",
            "img_path": save_image(question_id, annotation["images"], output_dir),
        }
        img_prefix = ""
        for i in range(len(item["img_path"])):
            img_prefix += f"<image {i + 1}> "
        item["question"] = img_prefix + question
        content.append(item)
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
