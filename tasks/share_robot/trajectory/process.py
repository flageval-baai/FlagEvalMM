import json
import os
import os.path as osp

import tqdm
from datasets import load_dataset
from PIL import Image


def process(cfg):

    data_dir, split = cfg.dataset_path, cfg.split
    output_root = cfg.processed_dataset_path
    output_dir = osp.join(output_root, split)
    os.makedirs(output_dir, exist_ok=True)

    cmd = f"huggingface-cli download --repo-type dataset --resume-download {data_dir} --local-dir {os.path.dirname(os.path.dirname(output_root))}"
    os.system(cmd)

    dataset = load_dataset(os.path.dirname(output_root), split=split)

    processed_data = []
    for i, item in tqdm.tqdm(enumerate(dataset)):
        conversations = item["conversations"][:2]
        question, answer = "", ""
        for conv in conversations:
            if conv["from"] == "human":
                question = conv["value"]
            if conv["from"] == "gpt":
                answer = conv["value"]
        if not question or not answer:
            continue
        image_path = osp.join(os.path.dirname(output_root), item["image"][0])
        image = Image.open(image_path)
        width, height = image.size
        processed_item = {
            "question_id": item["id"],
            "img_path": image_path,
            "question": question.replace(
                "Please predict up to 10 key trajectory points to complete the task.",
                "Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point.",
            ).replace("<image>", ""),
            "answer": answer,
            "question_type": "short-answer",
            "width": width,
            "height": height,
        }
        processed_data.append(processed_item)

    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(processed_data, f, indent=4)
