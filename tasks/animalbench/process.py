import json
import os
from pathlib import Path
import os.path as osp
from typing import List
import glob
import subprocess
import zipfile

# Video prefix mapping for different sub-tasks.
# Map JSON stem -> local video folder (relative to data_root).
JSON_TO_LOCAL_DIR = {
    "action_count": "TGIF-QA",
    "action_localization": "animal_kingdom/video_grounding",
    "action_prediction": "animal_kingdom/video_grounding",
    "action_sequence": "animal_kingdom/video_grounding",
    "AK_action_recognition": "animal_kingdom/video",
    "AK_bm": "animal_kingdom/video",
    "AK_object_recognition": "animal_kingdom/video",
    "AK_pd": "animal_kingdom/video",
    "AK_pm": "animal_kingdom/video",
    "AK_sa": "animal_kingdom/video",
    "LoTE_bm": "LoTE-Animal",
    "LoTE_sa": "LoTE-Animal",
    "mmnet_action_recognition": "mmnet",
    "mmnet_bm": "mmnet",
    "mmnet_object_recognition": "mmnet",
    "mmnet_pd": "mmnet",
    "mmnet_pm": "mmnet",
    "mmnet_sa": "mmnet",
    "object_count": "MSRVTT-QA",
    "object_existence": "mmnet",
    "reasoning": "NExT-QA",
}

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]


def download_dataset(repo_id, output_dir):
    """Download dataset from HuggingFace"""
    command = f"huggingface-cli download {repo_id} --repo-type dataset --local-dir {output_dir}"
    print(command)
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True
        )
        print(f"Download successful: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e.stderr}")
        # For now, skip download if it fails (using local data)
        print("Skipping download, using local data if available")


def extract_video_zips(output_dir):
    """Extract all zip files in videos/ directory

    Returns:
        set: Set of successfully extracted folder names
    """
    videos_dir = osp.join(output_dir, "videos")
    if not osp.exists(videos_dir):
        print(f"Videos directory not found at {videos_dir}")
        return set()

    zip_files = glob.glob(osp.join(videos_dir, "*.zip"))
    if not zip_files:
        print("No zip files found to extract")
        return set()

    successful_extracts = set()

    for zip_path in zip_files:
        zip_name = Path(zip_path).stem  # e.g., "TGIF-QA"
        extract_to = osp.join(videos_dir, zip_name)

        # Skip if already extracted
        if osp.exists(extract_to) and os.listdir(extract_to):
            print(f"Already extracted: {zip_name}")
            successful_extracts.add(zip_name)
            continue

        print(f"Extracting {zip_name}.zip...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # First, extract to a temporary location
                temp_extract = osp.join(videos_dir, f"_temp_{zip_name}")
                os.makedirs(temp_extract, exist_ok=True)

                # Extract all files
                zip_ref.extractall(temp_extract)

                # Check if there's a single top-level directory with the same name
                temp_contents = os.listdir(temp_extract)

                if len(temp_contents) == 1 and osp.isdir(
                    osp.join(temp_extract, temp_contents[0])
                ):
                    # If zip contains a single directory, move its contents up one level
                    inner_dir = osp.join(temp_extract, temp_contents[0])
                    if temp_contents[0] == zip_name or temp_contents[0] in [
                        zip_name.replace("-", "_"),
                        zip_name.replace("_", "-"),
                    ]:
                        # Same name detected, move contents directly
                        print(
                            f"  Detected nested '{temp_contents[0]}' directory, flattening..."
                        )
                        os.rename(inner_dir, extract_to)
                    else:
                        # Different name, keep as is
                        os.rename(temp_extract, extract_to)
                else:
                    # Multiple items at top level, keep structure as is
                    os.rename(temp_extract, extract_to)

                # Clean up temp directory if it still exists
                if osp.exists(temp_extract):
                    import shutil

                    shutil.rmtree(temp_extract)

                print(f"  Successfully extracted to {extract_to}")
                successful_extracts.add(zip_name)

        except zipfile.BadZipFile as e:
            print(f"  ✗ ERROR: Bad zip file {zip_path}: {e}")
            print(
                f"  → Skipping {zip_name}, related data will be excluded from evaluation"
            )
        except Exception as e:
            print(f"  ✗ ERROR: Failed to extract {zip_path}: {type(e).__name__}: {e}")
            print(
                f"  → Skipping {zip_name}, related data will be excluded from evaluation"
            )
            # Clean up any partial extraction
            temp_extract = osp.join(videos_dir, f"_temp_{zip_name}")
            if osp.exists(temp_extract):
                import shutil

                shutil.rmtree(temp_extract)

    return successful_extracts


def process_single_json(json_file, video_prefix) -> List[dict]:
    """Process a single JSON file and convert to standard format"""
    sub_task = Path(json_file).stem
    processed_data = []

    with open(json_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    for ann in annotations:
        raw_answer = ann["answer"]
        options = ann["candidates"]

        # Find answer index in candidates
        try:
            answer_index = options.index(raw_answer)
        except ValueError:
            # If answer not found in candidates, skip this item
            print(
                f"Warning: Answer '{raw_answer}' not found in candidates for video {ann.get('video', 'unknown')}"
            )
            continue

        # Build video path: video_prefix + video filename
        video_filename = ann["video"]
        video_path = osp.join(video_prefix, video_filename)

        processed_data.append(
            {
                "question": ann["question"],
                "raw_answer": raw_answer,
                "answer": (
                    CHOICE_LABELS[answer_index]
                    if answer_index < len(CHOICE_LABELS)
                    else str(answer_index)
                ),
                "options": options,
                "sub_task": sub_task,
                "question_type": "multiple-choice",
                "video_path": video_path,
            }
        )

    return processed_data


def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    # build output path
    # If name is empty, output_dir = {processed_dataset_path}/{split}
    # Otherwise, output_dir = {processed_dataset_path}/{name}/{split}
    if name:
        output_dir = osp.join(cfg.processed_dataset_path, name, split)
    else:
        output_dir = osp.join(cfg.processed_dataset_path, split)
    os.makedirs(output_dir, exist_ok=True)

    # Download dataset
    download_dataset(data_dir, output_dir)

    # Extract video zip files
    extract_video_zips(output_dir)

    all_processed_data = []
    skipped_tasks = []

    # Process each JSON file in the directory
    json_dir = osp.join(output_dir, "json")

    # Check if json_dir exists, if not, try to use local data directory
    if not osp.exists(json_dir):
        # Try to use local Animal-Bench/data directory if available
        local_data_dir = osp.join(
            osp.dirname(osp.dirname(osp.dirname(__file__))), "Animal-Bench", "data"
        )
        if osp.exists(local_data_dir):
            print(f"Using local data directory: {local_data_dir}")
            json_dir = local_data_dir
        else:
            print(f"Warning: JSON directory not found at {json_dir}")
            print("Please ensure data is downloaded or provide correct path")
            return

    json_files = glob.glob(osp.join(json_dir, "*.json"))

    if not json_files:
        print(f"Warning: No JSON files found in {json_dir}")
        return

    for json_path in json_files:
        filename = os.path.basename(json_path)
        json_stem = Path(json_path).stem
        local_dir = JSON_TO_LOCAL_DIR.get(json_stem, "")
        video_prefix = osp.join("videos", local_dir) if local_dir else "videos"

        # Check if required video directory exists
        video_dir_check = osp.join(output_dir, video_prefix)
        if not osp.exists(video_dir_check):
            print(f"⚠ Skipping {filename}: video directory not found at {video_prefix}")
            skipped_tasks.append(json_stem)
            continue

        print(f"Processing {filename} with video prefix: {video_prefix}")
        processed = process_single_json(json_path, video_prefix)
        all_processed_data.extend(processed)
        print(f"  Processed {len(processed)} items from {filename}")

    # Add question_id to each entry
    for i, entry in enumerate(all_processed_data):
        entry["question_id"] = i

    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_processed_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Processed {len(all_processed_data)} items in total.")
    print(f"Data saved to {output_file}")

    if skipped_tasks:
        print(
            f"\n⚠ Skipped {len(skipped_tasks)} tasks due to missing video directories:"
        )
        for task in skipped_tasks:
            print(f"  - {task}")
        print("\nThese tasks will not be included in the evaluation.")

    print(f"{'='*60}")
