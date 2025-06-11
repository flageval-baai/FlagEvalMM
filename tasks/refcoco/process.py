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

    question_id_set = set()
    # process each item
    for annotation in data:
        question_id = annotation["question_id"]
        if question_id not in question_id_set:
            question_id_set.add(question_id)
        else:
            continue
        image_width, image_height = (
            annotation["image"].width,
            annotation["image"].height,
        )
        img_name = f"img/{annotation['file_name']}"
        # save image
        try:
            image = annotation["image"].convert("RGB")
            image_save_path = f"{output_dir}/{img_name}"
            image.save(image_save_path)
        except Exception as e:
            print(f"Error saving image {question_id}: {e}")

        bbox = [
            round(annotation["bbox"][0] / image_width, 2),
            round(annotation["bbox"][1] / image_height, 2),
            round((annotation["bbox"][0] + annotation["bbox"][2]) / image_width, 2),
            round((annotation["bbox"][1] + annotation["bbox"][3]) / image_height, 2),
        ]

        # currently, the dataset has `answer` as a list of strings
        # each answer should be its own row
        # we will explode the dataset to have one row per answer
        # duplicate the other columns
        for index, answer in enumerate(annotation["answer"]):
            question = "Please generate a set of bounding box (bbox) coordinates based on the image and description.The bbox coordinate format is (top-left x, top-left y, bottom-right x, bottom-right y).All values must be floating-point numbers between 0 and 1, inclusive. For example, a valid bbox might be (0.1, 0.2, 0.5, 0.6)."
            # build information dictionary
            info = {
                "question": f"{question} Description: {answer}",
                "answer": bbox,
                "question_id": f"{question_id}_{index}",
                "img_path": img_name,
                "image_width": image_width,
                "image_height": image_height,
                "question_type": "bbox",
            }
            content.append(info)

    # save data
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
