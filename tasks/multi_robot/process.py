
import re
import os
import tqdm
import json
import os.path as osp
from datasets import load_dataset


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split

    with open(f"{data_dir}/data.json", "r") as f:
        data = json.load(f)  
    
    output_dir = osp.join(cfg.processed_dataset_path, cfg.split, cfg.get("dataset_name", ""))
    os.makedirs(output_dir, exist_ok=True)
    

    processed_data = []
    for i, item in enumerate(data):
        conversations = item["conversations"]
        question, answer = "", ""
        for conv in conversations:
            if conv["from"] == "human":
                question = conv["value"]
            if conv["from"] == "gpt":
                answer = conv["value"]
        if  not question or not answer:
            continue
        processed_item = {
            "question_id": item["id"],
            "question": question,
            "answer": answer,
            "question_type": "short-answer",
            "img_path": ""
        }
        processed_data.append(processed_item)

    with open(osp.join(output_dir, f"data.json"), "w") as f:
        json.dump(processed_data, f, indent=4)
        
