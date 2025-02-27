from typing import Optional, Any, Dict, List, Union
import os.path as osp
import json
from torch.utils.data import Dataset
from flagevalmm.registry import DATASETS, PROMPTS
from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR
from flagevalmm.common.logger import get_logger
from flagevalmm.dataset.utils import get_data_root

logger = get_logger(__name__)


@DATASETS.register_module()
class VqaBaseDataset(Dataset):
    def __init__(
        self,
        *,
        name: str,
        data_root: Optional[str] = None,
        anno_file: Optional[Union[str, List[str]]] = None,
        cache_dir: str = FLAGEVALMM_DATASETS_CACHE_DIR,
        config: Optional[dict] = None,
        prompt_template: Optional[str] = None,
        base_dir: Optional[str] = None,
        debug: bool = False,
        with_label: bool = False,
    ) -> None:
        self.data_root = get_data_root(
            data_root=data_root, config=config, cache_dir=cache_dir, base_dir=base_dir
        )

        anno_file = "data.json" if anno_file is None else anno_file
        self.annotations = self.load_annotations(anno_file)
        self.name = name
        if prompt_template is not None:
            self.prompt_template = PROMPTS.build(prompt_template)
        else:
            self.prompt_template = None
        self.with_label = with_label or debug
        if debug:
            self.annotations = self.annotations[:32]

    def load_annotations(self, anno_file: Union[str, List[str]]):
        if isinstance(anno_file, str):
            annotations = json.load(open(osp.join(self.data_root, anno_file)))
        else:
            annotations = []
            for file in anno_file:
                annotations.extend(json.load(open(osp.join(self.data_root, file))))
        return annotations

    def __len__(self) -> int:
        return len(self.annotations)

    def build_prompt(self, annotation: Dict[str, Any], img_path: List[str]) -> str:
        question: str = annotation["question"]
        choices = annotation.get("options", [])
        base = ord("A")
        for i, choice in enumerate(choices):
            question += "\n" + chr(base + i) + ". " + choice
        if len(img_path) > 0 and "<image 1>" not in question:
            question = "<image 1> " + question
        if self.prompt_template is not None:
            question = self.prompt_template.build_prompt(
                question=question,
                question_type=annotation["question_type"],
                subject=annotation.get("subject", None),
            )
        return question

    def __getitem__(self, index: int) -> Dict[str, Any]:
        annotation = self.annotations[index]
        img_path = []
        if annotation["img_path"] is not None:
            if isinstance(annotation["img_path"], list):
                for path in annotation["img_path"]:
                    img_path.append(osp.join(self.data_root, path))
            elif annotation["img_path"] is not None:
                img_path.append(osp.join(self.data_root, annotation["img_path"]))
        ret = {
            "img_path": img_path,
            "question": self.build_prompt(annotation, img_path),
            "question_id": str(annotation["question_id"]),
            "type": annotation["question_type"],
        }
        if self.with_label:
            ret["label"] = annotation["answer"]
        return ret

    def meta_info(self) -> Dict[str, Any]:
        return {"name": self.name, "length": len(self.annotations), "type": "vqa"}

    def get_annotation(self) -> Dict[str, Dict[str, Any]]:
        anno_dict = {}  # question_id: answer_dict
        for anno in self.annotations:
            anno_dict[str(anno["question_id"])] = anno
        return anno_dict
