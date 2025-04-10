import os.path as osp
from typing import Dict, Any
from flagevalmm.registry import DATASETS
from flagevalmm.dataset.vqa_base_dataset import VqaBaseDataset


@DATASETS.register_module()
class VideoDataset(VqaBaseDataset):

    def __getitem__(self, index: int) -> Dict[str, Any]:
        annotation = self.annotations[index]
        ret = {
            "video_path": osp.join(self.data_root, annotation["video_path"]),
            "question": self.build_prompt(annotation, []),
            "question_id": str(annotation["question_id"]),
            "type": annotation["question_type"],
        }
        if self.with_label and "answer" in annotation:
            ret["label"] = annotation["answer"]
        return ret

    def meta_info(self) -> Dict[str, Any]:
        return {"name": self.name, "length": len(self.annotations), "type": "video_qa"}