import json
import os
import os.path as osp

from datasets import load_dataset

need_to_replace = [
    "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points.",
    "Answer yes or no.",
]


def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")
    splits = ["compatibility", "configuration", "context"]
    question_type_map = {
        "compatibility": "yes-no",
        "configuration": "yes-no",
        "context": "point",
    }
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    img_dir = osp.join(output_dir, "img")
    # mask_dir = osp.join(output_dir, "mask")

    os.makedirs(img_dir, exist_ok=True)
    # os.makedirs(mask_dir, exist_ok= True)
    content = []
    for split in splits:
        data = load_dataset(data_dir, name=name, split=split)
        for index, item in enumerate(data):
            question_id = f"{split}_{index}"
            image_path = f"img/{question_id}.jpg"

            item["img"].save(osp.join(output_dir, image_path))
            # if split == "context":
            #     mask_path = f"mask/{question_id}.jpg"
            #     item["mask"].save(osp.join(output_dir, mask_path))
            question = item["question"]
            for s in need_to_replace:
                question = question.replace(s, "").strip()
            info = {
                "question_id": question_id,
                "question": question,
                "category": item["category"],
                "answer": item["answer"],
                "question_type": question_type_map[split],
                "img_path": image_path,
                "image_width": item["img"].width,
                "image_height": item["img"].height,
            }
            content.append(info)
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
