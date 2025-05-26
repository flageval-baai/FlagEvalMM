from datasets import load_dataset
import os.path as osp
import os
import json
import tqdm
import argparse

def process(cfg):
    print(cfg)
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    dataset = load_dataset(data_dir, split=split)
    content = []
    os.makedirs(osp.join(output_dir, "img"), exist_ok=True)
    for i, annotation in tqdm.tqdm(enumerate(dataset)):
        info = {
            "question_id": str(i),
            "question": annotation["question"],
            "answer": annotation["answer"],
            "question_type": annotation["question_type"],
            "dataset": annotation["dataset"],
        }
        image_path = f"img/{i}.png"
        annotation["image"].save(osp.join(output_dir, image_path))
        info["img_path"] = image_path
        content.append(info)
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(content, f, indent=2)



 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="textvqa", help="HuggingFace 数据集名称或路径")
    parser.add_argument("--split", type=str, default="train", help="数据集分片（train/val/test）")
    parser.add_argument("--processed_dataset_path", type=str, default="./processed_data", help="处理后的数据存储路径")
    return parser.parse_args()


if __name__ == "__main__":

    cfg = get_args()  # 从命令行解析参数
    process(cfg)
