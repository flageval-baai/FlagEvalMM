
import re
import os
import tqdm
import json
import os.path as osp
from datasets import load_dataset
from  PIL import Image


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    with open(f"{data_dir}/affordance_data_test.json", "r") as f:
        affordance = json.load(f)
    
    
    output_dir = osp.join(cfg.processed_dataset_path, cfg.split, cfg.get("dataset_name", ""))
    os.makedirs(output_dir, exist_ok=True)

    processed_data = []
    for i, item in enumerate(affordance):
        conversations = item["conversations"]
        question, answer = "", ""
        for conv in conversations:
            if conv["from"] == "human":
                question = conv["value"]
            if conv["from"] == "gpt":
                answer = conv["value"]
        if  not question or not answer:
            continue
        img_path =  item["image"]
        if  isinstance(img_path, list):
            img_path = img_path[0]
        image_path = osp.join(output_dir, img_path)
        image = Image.open(image_path)

        width, height = image.size

        processed_item = {
            "question_id": item["id"],
            "img_path": image_path,
            "question": question.replace("<image>", ""),
            "answer": answer,
            "question_type": "short-answer",
            "width": width,
            "height": height
        }
        processed_data.append(processed_item)

    with open(osp.join(output_dir, f"data.json"), "w") as f:
        json.dump(processed_data, f, indent=4)
        
