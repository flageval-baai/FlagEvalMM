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

    # process each item
    for index, annotation in enumerate(data):
        question_id = annotation["question_id"]
        # build information dictionary
        info = {
            "question_id": question_id,
            "answer": annotation["answer"],
            "sub_task": annotation["question_type"],
            "question_type": "multiple-choice",
            "img_path": [],
        }

        question = annotation["question"].replace(
            " Please answer directly with only the letter of the correct option and nothing else.",
            "",
        )
        images = annotation["images"]
        visual_indices = annotation.get("visual_indices", [])

        # save img
        for i, image in enumerate(images):
            image_path = osp.join("img", f"{question_id}_{i + 1}.jpg")
            info["img_path"].append(image_path)
            full_image_path = osp.join(output_dir, image_path)
            os.makedirs(osp.dirname(full_image_path), exist_ok=True)
            try:
                image.save(full_image_path)
            except Exception as e:
                print(f"Error saving image {question_id}: {e}")

        # contents: mix images and text according to visual_indices
        if not visual_indices or all(idx == 0 for idx in visual_indices):
            contents = []
            for i in range(len(images)):
                contents.append(f"<image {i + 1}>")
            contents.append(question)
        else:
            # Sort by visual_indices and insert images into the specified text positions
            pairs = list(zip(range(len(images)), visual_indices))
            pairs.sort(key=lambda x: x[1])
            last_pos = 0
            contents = []
            for img_idx, idx in pairs:
                if idx > last_pos:
                    contents.append(question[last_pos:idx])
                    contents.append(f"<image {img_idx + 1}>")
                    last_pos = idx
                elif idx == 0:
                    contents.append(f"<image {img_idx + 1}>")
                else:
                    contents.append(f"<image {img_idx + 1}>")
            if last_pos < len(question):
                contents.append(question[last_pos:])
            if not contents:
                for i in range(len(images)):
                    contents.append(f"<image {i + 1}>")
                contents.append(question)

        info["question"] = "".join(contents).strip()
        content.append(info)

    # save data
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
