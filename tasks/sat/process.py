import json
import os
import os.path as osp
from typing import List

from datasets import load_dataset

options = ["A", "B", "C", "D", "E", "F"]


def format_options(items):
    if len(items) > len(options):
        raise ValueError("Options not in A-F. Please check the number of options. ")
    formatted_options = "\n".join(
        f"{options[i]}. {item}" for i, item in enumerate(items)
    )
    return f"Options:\n{formatted_options}"


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
        question = annotation["question"]
        info = {
            "question_id": annotation["question_id"],
            "sub_task": annotation["question_type"],
            "question_type": "multiple-choice",
            "img_path": [],
        }
        answers = annotation["answers"]
        info["options"] = answers
        correct_answer = annotation["correct_answer"]
        answer_index = answers.index(correct_answer)
        info["answer"] = options[answer_index]
        info["img_path"] = save_image(
            annotation["question_id"], annotation["images"], output_dir
        )
        img_prefix = ""
        for i in range(len(info["img_path"])):
            img_prefix += f"<image {i + 1}> "
        info["question"] = img_prefix + question
        content.append(info)
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
