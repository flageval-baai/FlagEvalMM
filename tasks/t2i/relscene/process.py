import json
import os
import os.path as osp
from datasets import load_dataset


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    dataset = load_dataset(data_dir, name=name)["train"]
    all_descriptions = []
    os.makedirs(output_dir, exist_ok=True)
    for i, annotation in enumerate(dataset):
        if "long_description" in annotation:
            description = annotation["long_description"]
            all_descriptions.append({"id": i, "prompt": description})

    json.dump(all_descriptions, open(osp.join(output_dir, "data.json"), "w"), indent=2)


if __name__ == "__main__":
    from easydict import EasyDict

    config = {
        "dataset_path": "fierytrees/RelScene",
        "split": "train",
        "processed_dataset_path": osp.join(osp.expanduser("~"), "output/relscene"),
    }
    cfg = EasyDict(config)
    extracted_data = process(cfg)

# python flagevalmm/server/run_server.py --tasks tasks/t2i/relscene/relscene_test.py --output-dir /home/dengzijun/output --port 11825

# CUDA_VISIBLE_DEVICES=8,9 python flagevalmm/eval.py --output-dir /home/dengzijun/output --tasks tasks/t2i/relscene/relscene_test.py --model black-forest-labs/FLUX.1-schnell --exec model_zoo/t2i/flux/model_adapter.py --server-port 11825
