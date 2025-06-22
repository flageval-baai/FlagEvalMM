import json
from datasets import load_dataset
import os.path as osp
import os
import re


def extract_answer(answer_str):
    match = re.search(r'\((.*?)\)', answer_str)
    return match.group(1)


def save_image(image, filename, output_dir) -> str:
    image_path = osp.join("img", filename)
    full_image_path = osp.join(output_dir, image_path)
    os.makedirs(osp.dirname(full_image_path), exist_ok=True)
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(full_image_path)
    except Exception as e:
        print(f"Error saving image {filename}: {e}")
    return image_path


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    output_dir = osp.join(cfg.processed_dataset_path, name, split)

    content = []
    dataset = load_dataset(data_dir, split=split)
    for data in dataset:
        if data.get('type') == 'Robustness': # Skip Robustness type
            continue

        item = {
            "question_id": data["idx"],
            "question_type": "multiple-choice",
            "category": data["type"],
            "sub_task": data["task"],
            "img_path": save_image(data["image"], data["filename"], output_dir),
            "options": data['choices'],
            "question": data['question'],
            "answer": extract_answer(data['answer']),
        }
        content.append(item)

    # save data
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")