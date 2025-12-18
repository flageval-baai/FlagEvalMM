import json
import os
import os.path as osp
from datasets import load_dataset

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

    # process each item
    for index, annotation in enumerate(data):
        question_id = annotation["question_id"]
        question = annotation["question"]
        images = annotation["images"]
        # build information dictionary
        contents = []
        for i in range(len(images)):
            contents.append(f"<image {i + 1}>")
        contents.append(question)

        info = {
            "question_id": question_id,
            "question": "".join(contents).strip(),
            "img_path": [],
            "reference": annotation["reference"],
            "question_type": annotation["question_type"],
            "task_category": annotation["task_category"],
            "task_sub_category": annotation["task_sub_category"],
            "evaluator": annotation["evaluator"],
            "evaluator_kwargs": json.loads(annotation["evaluator_kwargs"])
        }
        for i, image in enumerate(images):
            image_path = osp.join("img", f"{question_id}_{i + 1}.png")
            info["img_path"].append(image_path)
            full_image_path = osp.join(output_dir, image_path)
            os.makedirs(osp.dirname(full_image_path), exist_ok=True)
            try:
                image.save(full_image_path)
            except Exception as e:
                print(f"Error saving image {question_id}: {e}")

        content.append(info)

    # save data
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
