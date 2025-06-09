import json
import os
import os.path as osp
from datasets import load_dataset


def save_image(image_path, image, output_dir):
    full_image_path = osp.join(output_dir, image_path)
    os.makedirs(osp.dirname(full_image_path), exist_ok=True)
    try:
        image.save(full_image_path)
    except Exception as e:
        print(f"Error saving image {image_path}: {e}")


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
        info = {
            "question_id": annotation["question_id"],
            "question": annotation["question"],
            "question_subtype": annotation["question_subtype"],
            "question_type": annotation["question_type"],
            "img_path": annotation["img_path"],
            "answer": eval(annotation["answer"]),
        }
        save_image(annotation["img_path"], annotation["image"], output_dir)
        content.append(info)
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
