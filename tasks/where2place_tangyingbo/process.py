import json
import os
import os.path as osp
from datasets import load_dataset
import base64
from PIL import Image


def save_image(full_image_path, image_obj):
    image_obj.save(full_image_path)


def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    # build output path
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    os.makedirs(output_dir, exist_ok=True)
    # img_dir = osp.join(output_dir, "img")
    # mask_idr = osp.join(output_dir, "mask")
    # os.makedirs(img_dir, exist_ok=True)
    # os.makedirs(mask_idr, exist_ok=True)

    # load dataset
    # data = load_dataset(data_dir, name=name, split=split)
    with open (f"{data_dir}/point_questions.jsonl", "r" )  as f:
        # data = json.load(f)
        data = f.readlines()
    content = []

    # process each item
    for index, annotation in enumerate(data):
        annotation = json.loads(annotation)
        question_id = annotation["question_id"]
        image_path = osp.join("/home/tangyingbo/dataset/where2place", f"images/{annotation['image']}" ) 
        mask_path = osp.join("/home/tangyingbo/dataset/where2place", f"mask/{annotation['image']}")
        # annotation["image"].save(osp.join(output_dir, image_path))
        # annotation["mask"].save(osp.join(output_dir, mask_path))
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"图片尺寸: {width} x {height}")

        # 转换为 Base64
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode('utf-8')
            print("Base64 编码已生成")
        # build information dictionary
        info = {
            "question_id": question_id,
            "question": f'<image 1> {annotation["text"]}',
            "sub_task": annotation["category"],
            "answer": mask_path,
            "question_type": "point",
            "img_path": image_path,
            "image_width": width,
            "image_height": height,
            "mask_path": mask_path,
            "mask_base64": base64_data,
        }
        content.append(info)

    # save data
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
