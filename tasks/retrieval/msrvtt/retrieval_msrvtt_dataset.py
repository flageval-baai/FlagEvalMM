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
from rawframe_util import RawFrameExtractor

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
        max_words: int = 30,
        feature_framerate: float = 1.0,
        max_frames: int = 100,
        image_resolution: int = 224,
        tokenizer=None,  # Add tokenizer parameter
        **kwargs,
    ) -> None:
        self.data_root = get_data_root(
            data_root=data_root, config=config, cache_dir=cache_dir, base_dir=base_dir
        )

        self.name = name
        # Load the MSRVTT dataset CSV file
        self.csv_path = os.path.join(self.data_root, anno_file if anno_file else "MSRVTT_JSFUSION_test.csv")
        self.data = pd.read_csv(self.csv_path)

        # Initialize the video frame extractor
        self.frameExtractor = RawFrameExtractor(framerate=feature_framerate, size=image_resolution)

        # Initialize text and video-related parameters
        self.max_words = max_words
        self.max_frames = max_frames
        self.feature_framerate = feature_framerate
        self.image_resolution = image_resolution
        self.tokenizer = tokenizer  # Set tokenizer

        # Load video and caption data
        self.videos = self.data['video_id'].values
        self.captions = self.data['sentence'].values

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        video_id = self.videos[index]
        caption = self.captions[index]

        # Get video frames
        video_frames, video_mask = self._get_rawvideo(video_id)

        # Get text features
        text_features, text_mask = self._get_text(caption)

        return video_frames, video_mask, text_features, text_mask

    def _get_rawvideo(self, video_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get video frame data"""
        video_mask = np.zeros((1, self.max_frames), dtype=np.long)
        video = np.zeros((1, self.max_frames, 1, 3, self.image_resolution, self.image_resolution), dtype=np.float)

        video_path = os.path.join(self.data_root, "MSRVTT_Videos", video_id + ".mp4")

        # Extract video frames
        raw_video_data = self.frameExtractor.get_video_data(video_path, self.max_frames)
        raw_video_data = raw_video_data['video']

        if len(raw_video_data.shape) > 3:
            raw_video_data_clip = raw_video_data
            raw_video_slice = self.frameExtractor.process_raw_data(raw_video_data_clip)

            if self.max_frames < raw_video_slice.shape[0]:
                sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                video_slice = raw_video_slice[sample_indx, ...]
            else:
                video_slice = raw_video_slice

            slice_len = video_slice.shape[0]
            video_mask[0][:slice_len] = [1] * slice_len
            video[0][:slice_len, ...] = video_slice

        else:
            print("get raw video error, skip it.")

        return video, video_mask

    def _get_text(self, caption: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get text features"""
        # Use tokenizer to encode text
        words = self.tokenizer.tokenize(caption)

        # Add CLS and SEP tokens
        words = [self.tokenizer.cls_token] + words + [self.tokenizer.sep_token]

        # Truncate or pad to the max length
        if len(words) > self.max_words:
            words = words[:self.max_words]
        else:
            words = words + [self.tokenizer.pad_token] * (self.max_words - len(words))

        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)

        # Pad to the max length
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)

        return np.array(input_ids), np.array(input_mask)

    def meta_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "video_number": len(self.videos),
            "caption_number": len(self.captions),
            "type": "retrieval",
        }
