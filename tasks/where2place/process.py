import json
import os
import os.path as osp
from datasets import load_dataset


def save_image(full_image_path, image_obj):
    image_obj.save(full_image_path)


def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    # build output path
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    img_dir = osp.join(output_dir, "img")
    mask_idr = osp.join(output_dir, "mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_idr, exist_ok=True)

    # load dataset
    data = load_dataset(data_dir, name=name, split=split)
    content = []

    # process each item
    for index, annotation in enumerate(data):
        question_id = annotation["question_id"]
        image_path = f"img/{question_id}.jpg"
        mask_path = f"mask/{question_id}.jpg"
        annotation["image"].save(osp.join(output_dir, image_path))
        annotation["mask"].save(osp.join(output_dir, mask_path))
        # build information dictionary
        info = {
            "question_id": question_id,
            "question": f'{annotation["question"]}',
            "sub_task": annotation["question_type"],
            "answer": mask_path,
            "question_type": "point",
            "img_path": image_path,
            "image_width": annotation["image"].width,
            "image_height": annotation["image"].height,
            "mask_path": mask_path,
        }
        content.append(info)

    # save data
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
