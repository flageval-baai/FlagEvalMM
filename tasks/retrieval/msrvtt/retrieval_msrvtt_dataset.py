from typing import Optional, Any, Dict, List, Tuple
import os
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from flagevalmm.registry import DATASETS
from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR
from flagevalmm.dataset.utils import get_data_root
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

@DATASETS.register_module()
class RetrievalMSRVTTDataset(Dataset):
    def __init__(
        self,
        *,
        name: str,
        data_root: Optional[str] = None,
        anno_file: Optional[str] = None,
        cache_dir: str = FLAGEVALMM_DATASETS_CACHE_DIR,
        config: Optional[dict] = None,
        base_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.data_root = get_data_root(
            data_root=data_root, config=config, cache_dir=cache_dir, base_dir=base_dir
        )

        self.name = name
        # Load the MSRVTT dataset CSV file
        self.csv_path = os.path.join(self.data_root, anno_file if anno_file else "MSRVTT_JSFUSION_test.csv")
        self.data = pd.read_csv(self.csv_path)

        # Load video and caption data
        self.videos = self.data['video_id'].values
        self.captions = self.data['sentence'].values

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        video_id = self.videos[index]
        video_path = os.path.join(self.data_root, "MSRVTT_Videos", video_id + ".mp4")
        caption = self.captions[index]
        return video_path, caption

    def get_video(self, video_index: int) -> Dict[str, str]:
        assert video_index < len(self)
        video_id = self.videos[video_index]
        return {"video_path": os.path.join(self.data_root, "MSRVTT_Videos", video_id + ".mp4")}

    def get_caption(self, catpion_index: int) -> Dict[str, str]:
        assert catpion_index < len(self)
        caption = self.captions[catpion_index]
        return {"caption": caption}

    def meta_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "video_number": len(self.videos),
            "caption_number": len(self.captions),
            "type": "retrieval",
        }

    def get_data(self, index: int, data_type: str) -> Dict[str, str]:
        if data_type == "video":
            return self.get_video(index)
        elif data_type == "text":
            return self.get_caption(index)
        else:
            raise Exception("Invalid data type")
