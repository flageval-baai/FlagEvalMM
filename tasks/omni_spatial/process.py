import os.path as osp
import json
import tqdm
import os
import zipfile

options = ["A", "B", "C", "D", "E", "F"]


def process(cfg):
    data_dir, split, anno_file = cfg.dataset_path, cfg.split, cfg.anno_file
    output_root = cfg.processed_dataset_path
    output_dir = osp.join(output_root, split)
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"huggingface-cli download --repo-type dataset --resume-download {data_dir} --local-dir {output_root}"
    os.system(cmd)
    data_json_file = osp.join(output_root, anno_file)
    dataset = json.load(open(data_json_file))
    content = []
    task_type_set = set()
    for index, annotation in tqdm.tqdm(enumerate(dataset)):
        task_type = annotation["task_type"]
        img_path = f"{task_type}/{annotation['id'].split('_')[0]}.png"
        info = {
            "question_id": index,
            "question": annotation["question"],
            "options": annotation["options"],
            "answer": options[annotation["answer"]],
            "question_type": "multiple-choice",
            "img_path": img_path,
            "category": task_type,
            "sub_task": annotation["sub_task_type"],
        }
        task_type_set.add(task_type)
        content.append(info)
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(content, f, indent=2)
    # unzip the datasets
    for dataset in task_type_set:
        zip_path = osp.join(output_root, f"{dataset}.zip")
        if osp.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
