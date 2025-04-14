import os
import os.path as osp
import json
from datasets import load_dataset


def process(cfg):
    data_dir = os.path.join(cfg.dataset_path, cfg.split)
    output_dir = osp.join(cfg.processed_dataset_path, cfg.split)
    os.makedirs(osp.join(output_dir, "img"), exist_ok=True)
    
    # Load validation parquet file
    data = load_dataset(data_dir, data_files={"train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]})
    content = []
    
    # Process each annotation in validation set
    for annotation in data['train']:
        info = {}

        # Extract fields from annotation
        info['question_id'] = f"visual_simpleqa_{str(annotation['id'])}"
        info['question'] = annotation['multimodal_question']
        info['answer'] = annotation['answer']
        info['question_type'] = "open-qa"
        info['rationale'] = annotation['rationale']
        info['image_source'] = annotation['image_source']
        info['evidence'] = annotation['evidence']
        info['category'] = annotation['category']
        info['rationale_granularity'] = annotation['rationale_granularity']

        if annotation['image'] is not None:
            img_name = f"img/{info['question_id']}.jpg"
            annotation['image'].save(osp.join(output_dir, img_name))
            info['img_path'] = img_name
        
        content.append(info)
    print(f'save {len(content)} annotations to {output_dir}')
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(content, f, indent=2)
