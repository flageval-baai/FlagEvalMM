import json
import os
import os.path as osp
from datasets import load_dataset
import zipfile
from huggingface_hub import hf_hub_download


def unzip(file_path, outpath):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(outpath)


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    data = load_dataset(data_dir, data_files="ucf_prompts_reformat.json")["train"]
    if not osp.exists(osp.join(output_dir, "video")):
        # Use official Hugging Face download function
        downloaded_file = hf_hub_download(
            repo_id=data_dir,
            filename="UCF101.zip",
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        os.makedirs(osp.join(output_dir, "video"), exist_ok=True)
        print("decompressing...")
        unzip(downloaded_file, osp.join(output_dir, "video"))

    content = []
    selected_keys = ["prompt", "id", "class_name"]
    for annotation in data:
        info = {k: annotation[k] for k in selected_keys}
        content.append(info)
    json.dump(content, open(osp.join(output_dir, "data.json"), "w"), indent=2)
