import json
import os
import os.path as osp
from datasets import load_dataset
import rarfile
from tqdm import tqdm
from flagevalmm.common.download_utils import download_file_with_progress


def unrar(file_path, outpath):
    rar_ = rarfile.RarFile(file_path)
    names = rar_.namelist()
    for name in tqdm(names):
        rar_.extract(name, outpath)
    rar_.close()



def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    data = load_dataset(data_dir,split="train")
    if not osp.exists(osp.join(output_dir, "video")):
        os.makedirs(osp.join(output_dir, "video"))
        path=osp.join(output_dir, "UCF101.rar")
        download_file_with_progress('https://huggingface.co/datasets/fierytrees/UCF/resolve/main/UCF101.rar?download=true',path)
        print('decompressing...')
        unrar(path,osp.join(output_dir, "video"))

    content = []
    selected_keys = ["prompt", "id", "class_name"]
    for annotation in data:
        info = {k: annotation[k] for k in selected_keys}
        content.append(info)
    json.dump(content, open(osp.join(output_dir, "data.json"), "w"), indent=2)


