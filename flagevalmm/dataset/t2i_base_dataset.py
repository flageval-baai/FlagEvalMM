import json
import os.path as osp
from typing import Dict, Optional
from torch.utils.data import Dataset
from flagevalmm.registry import DATASETS
from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR
from flagevalmm.dataset.utils import get_data_root


@DATASETS.register_module()
class Text2ImageBaseDataset(Dataset):
    def __init__(
        self,
        name: str,
        data_root: Optional[str] = None,
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
        self.data = json.load(open(osp.join(self.data_root, anno_file)))
        self.name = name
        self.debug = debug
        if debug:
            self.data = self.data[:32]

    def __len__(self) -> int:
        return len(self.data)

    def text_number(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return {"prompt": self.data[index]["prompt"], "id": str(self.data[index]["id"])}

    def get_data(self, index: int):
        assert index < self.text_number()
        return self.__getitem__(index)

    def meta_info(self):
        return {"name": self.name, "length": len(self.data), "type": "t2i"}

    def get_annotation(self):
        num = self.text_number()
        anno_dict = {}
        for i in range(num):
            cur_data = self.data[i]
            anno_dict[str(cur_data["id"])] = cur_data
        return anno_dict
