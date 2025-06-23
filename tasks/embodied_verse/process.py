import json
from PIL import Image
import time
import os.path as osp
from datasets import load_dataset
from huggingface_hub import snapshot_download

robo_post_prompt_point = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.

Each point should be represented as a normalized (x, y) tuple, where both x and y values are floats between 0 and 1, corresponding to the position within the image (e.g., for a point at pixel (50, 75) in a 100*100 image, the normalized coordinate is (0.5, 0.75)).

Format your final answer strictly as follows on the last line of your response:
Answer: [(x1, y1), (x2, y2), ..., (xn, yn)]

Do not include additional text after this line.
"""

robo_post_prompt_yes_no = """Your task is to answer the question above. Respond with a brief explanation if needed, followed by a yes or no answer in the last line of your response.

Format your final answer strictly as follows on the last line of your response:
Answer: yes or no

Do not include additional text after this line.
"""

where2place_post_prompt = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.

Each point should be represented as a normalized (x, y) tuple, where both x and y values are floats between 0 and 1, corresponding to the position within the image (e.g., for a point at pixel (50, 75) in a 100*100 image, the normalized coordinate is (0.5, 0.75)).

Format your final answer strictly as follows on the last line of your response:
Answer: [(x1, y1), (x2, y2), ..., (xn, yn)]

Do not include additional text after this line.
"""

PROMPT_MAP = {
    "SAT": "Carefully analyze the multiple-choice question above and reason through it step by step. Conclude your response with a line in the following format: Answer: $LETTER (without quotes), where $LETTER is the letter of the correct choice.",
    "erqa": "Carefully analyze the multiple-choice question above and reason through it step by step. Conclude your response with a line in the following format: Answer: $LETTER (without quotes), where $LETTER is the letter of the correct choice.",
    "Where2Place": where2place_post_prompt,
    "all_angles_bench": "Answer with the option's letter from the given choices directly.",
    "egoplan_bench2": "Answer with the option's letter from the given choices directly.",
    "cv_bench_test": "Answer with the option's letter from the given choices directly.",
    "embspatial_bench": "Answer with the option's letter from the given choices directly.",
    "blink_val_ev": "The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.",
    "vsi_bench_tiny": {
        "pre_prompt": "These are frames of a video.",
        "post_prompt": {
            "multiple-choice": "Carefully analyze the question above and reason through it step by step. Conclude your response with a line in the following format:\nAnswer: $LETTER (without quotes), where $LETTER corresponds to the correct option.",
            "numerical": "Carefully analyze the question above and reason through it step by step. Conclude your response with a line in the following format:\nAnswer: $NUMBER (without quotes), where $NUMBER is a number (integer or float) corresponds to the correct answer.",
        },
    },
    "robo_spatial_home_all": {
        "post_prompt": {
            "yes-no": robo_post_prompt_yes_no,
            "point": robo_post_prompt_point,
        }
    },
}


def add_prompt(ann_question, source, question_type):
    """Add prompt to the question based on the source"""
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


def download_media_files(repo_id: str, output_dir: str) -> bool:
    print(f"Download media files from {repo_id} dataset.")
    max_retries = 10
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=output_dir,
                max_workers=5,
            )
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return False


def get_image_dimensions(image_path) -> tuple:
    """Get the dimensions of an image file."""
    with Image.open(image_path) as img:
        # 获取并返回图片的尺寸 (宽度, 高度)
        width, height = img.size
        return width, height


def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    # build output path
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    output_file = osp.join(output_dir, "data.json")
    if not osp.exists(output_file):
        is_download_complete = download_media_files(data_dir, output_dir)
    else:
        is_download_complete = True
        print(
            f"Skipping download media files for {data_dir} dataset already complete exist."
        )
    if not is_download_complete:
        return
    data = load_dataset(data_dir, name=name, split=split)
    content = []

    # process each item
    for annotation in data:
        question = annotation["question"]
        source = annotation["source"]
        question_type = annotation["question_type"]
        item = {
            "question_id": annotation["question_id"],
            "raw_question_id": annotation["raw_question_id"],
            "level-1": annotation["level-1"],
            "level-2": annotation["level-2"],
            "question_type": question_type,
            "answer": annotation["answer"],
            "source": source,
        }
        if annotation.get("video_path"):
            item["video_path"] = "".join(annotation["video_path"])
        else:
            item["img_path"] = annotation.get("img_path")
        if question_type == "point":
            item["image_width"], item["image_height"] = get_image_dimensions(
                osp.join(output_dir, item["img_path"][0])
            )
            item["mask_path"] = annotation["mask_path"][0]
        question = format_options_optimized(question, annotation.get("options", []))
        question = add_prompt(question, source, question_type)
        item["question"] = question
        content.append(item)

    # save data
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")