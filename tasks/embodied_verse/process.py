import json
import time
import os.path as osp
import os
from datasets import load_dataset
from huggingface_hub import snapshot_download


def add_prompt(ann_question, source, question_type):
    """Add prompt to the question based on the source"""
    from tasks.embodied_verse.prompt import PROMPT_MAP

    prompt = PROMPT_MAP[source]
    if isinstance(prompt, str):
        question = f"{ann_question}\n{prompt}"
    elif isinstance(prompt, dict):
        pre_prompt = prompt.get("pre_prompt", "")
        post_prompt = prompt.get("post_prompt")[question_type]
        if pre_prompt:
            question = f"{pre_prompt}\n{ann_question}\n{post_prompt}"
        else:
            question = f"{ann_question}\n{post_prompt}"
    return question


def format_options_optimized(question: str, choices: list) -> str:
    if not choices:
        return question

    formatted_choices = [
        f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)
    ]
    return f"{question}\n" + "\n".join(formatted_choices)


def download_video_files(repo_id: str, output_dir: str) -> bool:
    print(f"Download video files from {repo_id} dataset.")
    max_retries = 10
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns="video/**",
                local_dir=output_dir,
            )
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return False


def save_mask_image(question_id, image, output_dir):
    mask_path = osp.join("img/mask", f"{question_id}.jpg")
    full_image_path = osp.join(output_dir, mask_path)
    os.makedirs(osp.dirname(full_image_path), exist_ok=True)
    try:
        if not osp.exists(full_image_path):
            image.save(full_image_path)
    except Exception as e:
        print(f"Error saving mask image {question_id}: {e}")
    return mask_path


def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    # build output path
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    output_file = osp.join(output_dir, "data.json")
    if not osp.exists(output_file):
        is_download_complete = download_video_files(data_dir, output_dir)
    else:
        is_download_complete = True
        print(
            f"Skipping download video files for {data_dir} dataset already complete exist."
        )
    if not is_download_complete:
        return
    data = load_dataset(data_dir, name=name, split=split)
    content = []

    # process each item
    for annotation in data:
        question_id = annotation["question_id"]
        source = annotation["source"]
        question_type = annotation["question_type"]
        item = {
            "question_id": annotation["question_id"],
            "raw_question_id": annotation["raw_question_id"],
            "question": annotation["question"],
            "level-1": annotation["level-1"],
            "level-2": annotation["level-2"],
            "question_type": question_type,
            "answer": annotation["answer"],
            "source": source,
        }
        if annotation.get("video_path"):
            item["video_path"] = annotation["video_path"]
        else:
            # save img
            item["img_path"] = []
            for i, image in enumerate(annotation["images"]):
                image_path = osp.join("img", f"{question_id}_{i + 1}.jpg")
                full_image_path = osp.join(output_dir, image_path)
                os.makedirs(osp.dirname(full_image_path), exist_ok=True)
                try:
                    if not osp.exists(full_image_path):
                        image.save(full_image_path)
                    item["img_path"].append(image_path)
                except Exception as e:
                    print(f"Error saving image {question_id}: {e}")
        if question_type == "point":
            item["image_width"], item["image_height"] = annotation["images"][0].size
            mask_path = save_mask_image(
                question_id, annotation["mask_image"], output_dir
            )
            item["mask_path"] = mask_path
            item["answer"] = mask_path
        content.append(item)

    # save data
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
