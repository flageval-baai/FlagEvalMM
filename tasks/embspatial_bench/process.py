import json
import os
import os.path as osp
import base64
from datasets import load_dataset


DIRECTION_SET = {"left", "right", "above", "under"}
DISTANCE_SET = {"close", "far"}
ANSWER_LETTERS = ["A", "B", "C", "D"]


def merge_relation(relation):
    """Map relationships to higher-level categories"""
    if relation in DIRECTION_SET:
        return "direction"
    elif relation in DISTANCE_SET:
        return "distance"
    else:
        raise ValueError(f"Unknown relation: {relation}")


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
    for annotation in data:
        question_id = annotation["question_id"]
        img_name = f"img/{question_id}.png"

        # build information dictionary
        info = {
            "question": annotation["question"],
            "answer": ANSWER_LETTERS[annotation["answer"]],
            "question_id": question_id,
            "img_path": img_name,
            "relation": annotation["relation"],
            "options": annotation["answer_options"],
            "sub_task": merge_relation(annotation["relation"]),
            "question_type": "multiple-choice",
        }

        # save image
        try:
            image_data = base64.b64decode(annotation["image"])
            with open(osp.join(output_dir, img_name), "wb") as image_file:
                image_file.write(image_data)
        except Exception as e:
            print(f"Error saving image {question_id}: {e}")

        # format question
        question = info["question"]
        if "options" in info:
            options_text = "\n".join([f"{opt}" for opt in info["options"]])
            question += f"\nOptions:\n{options_text}"

        info["formatted_question"] = question
        content.append(info)

    # save data
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
