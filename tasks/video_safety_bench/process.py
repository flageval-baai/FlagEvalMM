import json
import os
import os.path as osp
from datasets import load_dataset
import tarfile


def download_video(output_root, repo_id, hf_token):
    video_file_name = "video.tar.gz"
    local_filepath = osp.join(output_root, video_file_name)
    local_video_dirpath = osp.join(output_root, "video")
    if osp.exists(local_filepath):
        print(f"File '{local_filepath}' already exists locally. Skipping download.")
    else:
        print(f"File '{local_filepath}' not found. Starting download...")
        cmd_list = [
            "huggingface-cli",
            "download",
            repo_id,
            "--include",
            video_file_name,
            "--repo-type",
            "dataset",
            "--resume-download",
            "--local-dir",
            output_root,
            "--local-dir-use-symlinks",
            "False",
            "--token",
            hf_token,
        ]
        os.system(" ".join(cmd_list))
    if not osp.exists(local_video_dirpath) and osp.exists(local_filepath):
        with tarfile.open(local_filepath, "r:gz") as tar:
            tar.extractall(path=output_root)
    return local_video_dirpath


def process(cfg):
    """Process the dataset and save it in a standard format"""
    hf_token = os.environ.get("HF_TOKEN", "")
    assert (
        hf_token
    ), "HF_TOKEN is not set, please export HF_TOKEN=<your_huggingface_token>"
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    # build output path
    output_root = osp.join(cfg.processed_dataset_path, name)
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    os.makedirs(output_dir, exist_ok=True)
    # download videos
    download_video_dirpath = download_video(output_root, data_dir, hf_token)
    video_dir = osp.join(output_dir, name, "video")
    if not osp.lexists(video_dir):
        os.symlink(download_video_dirpath, video_dir, target_is_directory=True)

    # load dataset
    data = load_dataset(data_dir, name=name, split=split, token=hf_token)
    content = []

    # process each item
    for annotation in data:
        info = {
            "question_id": annotation["question_id"],
            "question": annotation["question"],
            "answer": "",
            "harmful_intention": annotation["harmful_intention"],
            "question_type": annotation["question_type"],
            "video_path": annotation["video_path"],
            "category": annotation["category"],
            "subcategory": annotation["subcategory"],
        }
        content.append(info)

    # save data
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
