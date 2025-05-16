import os.path as osp
import json
import shutil
import tqdm
import os
from PIL import Image

def load_dataset(cfg):
    json_path = osp.join(cfg.dataset_path, cfg.data_file)
    with open(json_path, "r") as f:
        dataset = json.load(f)
    return dataset


def process(cfg):
    dataset = load_dataset(cfg)

    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, cfg.split, name)
    os.makedirs(output_dir, exist_ok=True)
    content = []
    for i, data in enumerate(tqdm.tqdm(dataset)):
        new_data = {}
        new_data["question_id"] = data["question_id"]
        new_data["img_path"] = data["img_path"]
        new_data["question"] = data["question"]
        new_data["answer"] = data["answer"]
        new_data["question_type"] = data["data_type"]
        new_data["category"] = data.get("category", "")
        new_data["publication_date"] = data.get("publication_date", "")
        new_data["url"] = data.get("url", "")
        new_data["human_answer"] = data.get("human_answer", "")
        # structured_title = data.get("structured_title_raw", "")

        # Create the directory for the image if it doesn't exist
        os.makedirs(os.path.dirname(osp.join(output_dir, new_data["img_path"])), exist_ok=True)
        
        # Open and save image with error handling
        shutil.copy(os.path.join(cfg.dataset_path, data["img_path"]), osp.join(output_dir, new_data["img_path"]))
        content.append(new_data)
    print(f"process done, {len(content)} data, data.json saved in {output_dir}")
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(content, f, indent=4)
