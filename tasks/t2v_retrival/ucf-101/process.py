import json
import os
import os.path as osp
from datasets import load_dataset
import zipfile
from tqdm import tqdm
from flagevalmm.common.download_utils import download_file_with_progress


def unzip(file_path, outpath):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(outpath)



def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    data = load_dataset(data_dir,data_files='ucf_prompts_reformat.json')['train']
    if not osp.exists(osp.join(output_dir, "video")):
        os.makedirs(osp.join(output_dir, "video"))
        path=osp.join(output_dir, "UCF101.zip")
        download_file_with_progress('https://huggingface.co/datasets/fierytrees/UCF/resolve/main/UCF101.zip?download=true',path)
        print('decompressing...')
        unzip(path,osp.join(output_dir, "video"))

    content = []
    selected_keys = ["prompt", "id", "class_name"]
    for annotation in data:
        info = {k: annotation[k] for k in selected_keys}
        content.append(info)
    json.dump(content, open(osp.join(output_dir, "data.json"), "w"), indent=2)


