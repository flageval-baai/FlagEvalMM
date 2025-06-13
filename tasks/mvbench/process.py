import json
import os
from pathlib import Path
import os.path as osp
from typing import List
import glob
import subprocess

VIDEO_PREFIX_MAP = {
    "object_interaction.json": "star/Charades_segment",
    "action_sequence.json": "star/Charades_segment",
    "action_prediction.json": "star/Charades_segment",
    "action_localization.json": "sta/sta_video_segment",
    "moving_count.json": "clevrer/video_validation",
    "fine_grained_pose.json": "nturgbd_convert",
    "character_order.json": "perception/videos",
    "object_shuffle.json": "perception/videos",
    "egocentric_navigation.json": "vlnqa",
    "moving_direction.json": "clevrer/video_validation",
    "episodic_reasoning.json": "tvqa/video_fps3_hq_segment",
    "fine_grained_action.json": "Moments_in_Time_Raw/videos",
    "scene_transition.json": "scene_qa/video",
    "state_change.json": "perception/videos",
    "moving_attribute.json": "clevrer/video_validation",
    "action_antonym.json": "ssv2_video_mp4",
    "unexpected_action.json": "FunQA_test/test",
    "counterfactual_inference.json": "clevrer/video_validation",
    "object_existence.json": "clevrer/video_validation",
    "action_count.json": "perception/videos",
}

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]


def download_dataset(repo_id, output_dir):
    command = f"huggingface-cli download {repo_id} --repo-type dataset --revision video --local-dir {output_dir}"
    print(command)
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True
        )
        print(f"Download successful: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e.stderr}")


def process_single_json(json_file, video_prefix) -> List[dict]:
    sub_task = Path(json_file).stem
    processed_data = []

    with open(json_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    for ann in annotations:
        raw_answer = ann["answer"]
        options = ann["candidates"]
        answer_index = options.index(raw_answer)

        processed_data.append(
            {
                "question": ann["question"],
                "raw_answer": raw_answer,
                "answer": CHOICE_LABELS[answer_index],
                "options": options,
                "sub_task": sub_task,
                "question_type": "multiple-choice",
                "video_path": osp.join(video_prefix, ann["video"]),
            }
        )

    return processed_data


def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    # build output path
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    os.makedirs(output_dir, exist_ok=True)
    download_dataset(data_dir, output_dir)

    all_processed_data = []
    # Process each JSON file in the directory
    json_dir = osp.join(output_dir, "json")
    json_files = glob.glob(osp.join(json_dir, "*.json"))
    for json_path in json_files:
        filename = os.path.basename(json_path)
        video_prefix = VIDEO_PREFIX_MAP.get(filename)

        all_processed_data.extend(process_single_json(json_path, video_prefix))

    for i, entry in enumerate(all_processed_data):
        entry["question_id"] = i

    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(all_processed_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(all_processed_data)} items. Data saved to {output_file}")
