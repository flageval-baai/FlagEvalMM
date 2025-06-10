from typing import Optional, Any, Dict, List, Tuple
from torch.utils.data import Dataset
import json
import os
import os.path as osp
from flagevalmm.registry import DATASETS
from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR
from flagevalmm.dataset.utils import get_data_root
from numpy.random import randint


@DATASETS.register_module()
class VideoRetrievalDataset(Dataset):
    def __init__(
        self,
        *,
        name: str,
        data_root: Optional[str] = None,
        anno_file: Optional[str] = None,
        cache_dir: str = FLAGEVALMM_DATASETS_CACHE_DIR,
        caption_per_video: int = 1,
        config: Optional[dict] = None,
        base_dir: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        self.data_root = get_data_root(
            data_root=data_root, config=config, cache_dir=cache_dir, base_dir=base_dir
        )

        anno_file = "data.json" if anno_file is None else anno_file
        self.annotations = json.load(open(osp.join(self.data_root, anno_file)))
        self.name = name
        if debug:
            self.annotations = self.annotations[:160]
        # flatten the caption list

        self.captions = [annotation["prompt"] for annotation in self.annotations]

    def __getitem__(self, index: int) -> Tuple[str, List[str]]:
        root = self.data_root
        annotation = self.annotations[index]
        video_folder = os.path.join(root, "video", self.name, annotation["class_name"])
        lst = os.listdir(video_folder)
        video_path = os.path.join(video_folder, lst[randint(0, len(lst))])

        return video_path, annotation["caption"]

    def video_number(self) -> int:
        return len(self.annotations)

    def caption_number(self) -> int:
        return len(self.captions)

    def get_video(self, video_index: int) -> Dict[str, str]:
        assert video_index < self.video_number()
        annotation = self.annotations[video_index]
        video_folder = os.path.join(
            self.data_root, "video", self.name, annotation["class_name"]
        )
        lst = os.listdir(video_folder)
        video_path = os.path.join(video_folder, lst[randint(0, len(lst))])
        print(video_path)
        return {"video_path": video_path}

    def get_caption(self, catpion_index: int) -> Dict[str, str]:
        assert catpion_index < self.caption_number()
        caption = self.captions[catpion_index]
        return {"caption": caption}

    def meta_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "video_number": self.video_number(),
            "caption_number": self.caption_number(),
            "type": "retrieval",
        }

    def get_data(self, index: int, data_type: str) -> Dict[str, str]:
        if data_type == "video":
            return self.get_video(index)
        elif data_type == "text":
            return self.get_caption(index)
        else:
            raise Exception("Invalid data type")
