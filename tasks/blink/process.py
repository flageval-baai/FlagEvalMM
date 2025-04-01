from datasets import load_dataset
import os
import os.path as osp
import json
import tqdm


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    dataset_names = [
        "Art_Style",
        "Counting",
        "Forensic_Detection",
        "Functional_Correspondence",
        "IQ_Test",
        "Jigsaw",
        "Multi-view_Reasoning",
        "Object_Localization",
        "Relative_Depth",
        "Relative_Reflectance",
        "Semantic_Correspondence",
        "Spatial_Relation",
        "Visual_Correspondence",
        "Visual_Similarity",
    ]
    content = []
    max_image_num = 4
    output_base_dir = osp.join(cfg.processed_dataset_path, split)
    for dataset_name in dataset_names:
        dataset = load_dataset(data_dir, dataset_name, split=split)
        for data in tqdm.tqdm(dataset):
            new_data = {}
            new_data["question_id"] = data["idx"]
            new_data["question"] = data["prompt"]
            new_data["sub_task"] = data["sub_task"]
            new_data["answer"] = data["answer"][1]
            new_data["question_type"] = "multiple-choice"
            new_data["img_path"] = []
            for i in range(max_image_num):
                if data[f"image_{i + 1}"] is None:
                    break
                image_path = osp.join("image", f"{data['idx']}_{i + 1}.jpg")
                new_data["img_path"].append(image_path)
                full_image_path = osp.join(output_base_dir, image_path)
                os.makedirs(osp.dirname(full_image_path), exist_ok=True)
                data[f"image_{i + 1}"].save(full_image_path)
            img_prefix = ""
            for i in range(len(new_data["img_path"])):
                img_prefix += f"<image {i+1}> "
            new_data["question"] = img_prefix + new_data["question"]
            content.append(new_data)
    with open(osp.join(output_base_dir, "data.json"), "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=True)
