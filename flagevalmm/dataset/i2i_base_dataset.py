import json
import os.path as osp
from typing import Dict, Optional
from torch.utils.data import Dataset
from flagevalmm.registry import DATASETS
from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR
from flagevalmm.dataset.utils import get_data_root


@DATASETS.register_module()
class Image2ImageBaseDataset(Dataset):
    """Base dataset for image editing (I2I) benchmarks.

    Expected JSON schema per entry (keys are configurable):
      - source_key: path to the source image to edit
      - target_key: optional path to the reference/target image (if applicable)
      - prompt_key: edit instruction text
      - id: unique identifier
    """

    def __init__(
        self,
        name: str,
        data_root: Optional[str] = None,
        anno_file: Optional[str] = None,
        cache_dir: str = FLAGEVALMM_DATASETS_CACHE_DIR,
        config: Optional[dict] = None,
        base_dir: Optional[str] = None,
        debug: bool = False,
        source_key: str = "source",
        target_key: str = "target",
        prompt_key: str = "prompt",
        id_key: str = "id",
        image_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.data_root = get_data_root(
            data_root=data_root, config=config, cache_dir=cache_dir, base_dir=base_dir
        )
        self.image_root = (
            osp.join(self.data_root, image_dir) if image_dir is not None else self.data_root
        )
        self.source_key = source_key
        self.target_key = target_key
        self.prompt_key = prompt_key
        self.id_key = id_key

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

    def _resolve_path(self, rel_or_abs: str) -> str:
        if osp.isabs(rel_or_abs):
            return rel_or_abs
        return osp.join(self.image_root, rel_or_abs)

    def __getitem__(self, index: int) -> Dict:
        record = self.data[index]
        if self.source_key not in record:
            raise KeyError(f"Missing '{self.source_key}' in data index {index}")
        if self.prompt_key not in record:
            raise KeyError(f"Missing '{self.prompt_key}' in data index {index}")
        if self.id_key not in record:
            raise KeyError(f"Missing '{self.id_key}' in data index {index}")

        source_path = self._resolve_path(record[self.source_key])
        target_value = record.get(self.target_key)
        target_path = (
            self._resolve_path(target_value) if target_value is not None else None
        )

        return {
            "source_path": source_path,
            "target_path": target_path,
            "prompt": record[self.prompt_key],
            "id": str(record[self.id_key]),
        }

    def get_data(self, index: int):
        assert index < self.text_number()
        return self.__getitem__(index)

    def meta_info(self):
        return {"name": self.name, "length": len(self.data), "type": "i2i"}

    def get_annotation(self):
        num = self.text_number()
        anno_dict = {}
        for i in range(num):
            cur_data = dict(self.data[i])
            if self.id_key not in cur_data:
                raise KeyError(f"Missing '{self.id_key}' in data index {i}")

            id_value = cur_data.get(self.id_key)
            id_str = str(id_value)
            cur_data["id"] = id_str

            # Attach resolved absolute paths for convenience.
            if self.source_key in cur_data:
                cur_data["source_path"] = self._resolve_path(cur_data[self.source_key])
            target_value = cur_data.get(self.target_key)
            if target_value is not None:
                cur_data["target_path"] = self._resolve_path(target_value)

            anno_dict[id_str] = cur_data
        return anno_dict
