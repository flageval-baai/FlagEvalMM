import json
import os
import os.path as osp
import glob
import subprocess
from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR


def process(cfg):
    download_and_extract_dataset(
        cfg.dataset_path, cfg.processed_dataset_path, cfg.split
    )


def download_and_extract_dataset(repo_id, processed_dataset_path, split):
    cache_dir = osp.join(FLAGEVALMM_DATASETS_CACHE_DIR, processed_dataset_path, split)
    os.makedirs(cache_dir, exist_ok=True)

    command = f"huggingface-cli download {repo_id} --repo-type dataset --revision video --local-dir {cache_dir}"
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True
        )
        print("Download successful!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Download failed!")
        print(e.stderr)
        return

    json_dir = osp.join(cache_dir, "json")
    ann_file = osp.join(cache_dir, "data.json")
    video_prefix_mapping = {
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
        "action_recognition.json": "video/ActionRecog/",
    }
    batch_process_json_files(json_dir, video_prefix_mapping, ann_file)


def process_json(input_file, video_prefix):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_type = os.path.splitext(os.path.basename(input_file))[0]

    processed_data = []
    for annotation in data:
        answer = annotation["answer"]
        options = annotation["candidates"]
        if answer in options:
            answer_index = options.index(answer)
            answer_label = chr(ord("A") + answer_index) + " " + answer
        base = ord("A")
        question = annotation["question"]
        question = (
            "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
            + question
        )
        for i, choice in enumerate(options):
            question += "\n" + "(" + chr(base + i) + ")" + choice
        question += "\nOnly give the best option. \nBest option:("
        processed_entry = {
            "question": question,
            "answer": answer_label,
            "question_type": question_type,
            "video_path": osp.join(video_prefix, annotation["video"]),
        }
        processed_data.append(processed_entry)

    return processed_data


def batch_process_json_files(input_folder, video_prefix_mapping, output_file):
    all_processed_data = []
    global_question_id = 0

    input_files = glob.glob(osp.join(input_folder, "*.json"))

    for input_file in input_files:
        filename = os.path.basename(input_file)
        video_prefix = video_prefix_mapping.get(filename, "default_prefix")
        print(f"Processing {filename} with video prefix: {video_prefix}")

        processed_data = process_json(input_file, video_prefix)
        for entry in processed_data:
            entry["question_id"] = global_question_id
            global_question_id += 1

        all_processed_data.extend(processed_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_processed_data, f, indent=2)
