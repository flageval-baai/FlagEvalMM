import json
from typing import Any, Dict, Optional
from torch.utils.data import Dataset
from flagevalmm.registry import DATASETS
from flagevalmm.dataset.utils import get_data_root
from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR
import os.path as osp


@DATASETS.register_module()
class Text2VideoBaseDataset(Dataset):
    def __init__(
        self, name: str, data_root: Optional[str] = None,
        anno_file: Optional[str] = None,
        cache_dir: str = FLAGEVALMM_DATASETS_CACHE_DIR,
        config: Optional[dict] = None,
        base_dir: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        self.data_root = get_data_root(
            data_root=data_root, config=config, cache_dir=cache_dir, base_dir=base_dir
        )
        anno_file = "data.json" if anno_file is None else anno_file
        self.files = json.load(open(osp.join(self.data_root, anno_file)))
        self.name = name
        # self.files = json.load(open(data_root))
        self.debug = debug
        if debug:
            self.files = self.files[:32]

    def __len__(self):
        return len(self.files)

    def text_number(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "prompt": self.files[index]["prompt"],
            "id": str(self.files[index]["id"]),
        }

    def get_data(self, index: int) -> Dict[str, Any]:
        assert index < self.text_number()
        return self.__getitem__(index)

    def meta_info(self) -> Dict[str, Any]:
        return {"name": self.name, "length": len(self.files), "type": "t2v"}

    def get_annotation(self) -> Dict[str, Dict[str, Any]]:
        num = self.text_number()
        anno_dict = {}
        for i in range(num):
            data = self.get_data(i)
            anno_dict[data["id"]] = data
        return anno_dict
