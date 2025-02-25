from typing import Optional, Any, Dict, List, Tuple
from torch.utils.data import Dataset
import json
import os
import os.path as osp
from PIL import Image
from flagevalmm.registry import DATASETS
from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR
from flagevalmm.dataset.utils import get_data_root
import pandas as pd
import numpy as np
from rawframe_util import RawFrameExtractor


@DATASETS.register_module()
class MSRVTT_single_sentence_dataLoader(Dataset):
    """MSRVTT dataset loader for single sentence with integration to FlagEvalMM

    Attributes:
        name: dataset name
        data_root: path to the dataset root
        anno_file: annotation file (default is data.json)
        csv_path: path to the CSV file containing video id and sentence mapping
        features_path: path to the directory containing video frames
        tokenizer: tokenizer for text processing
        max_words: maximum number of words for tokenization
        feature_framerate: frame rate for sampling video
        max_frames: maximum number of frames to sample from each video
        image_resolution: resolution of images
        debug: whether to load a small subset for debugging
    """

    def __init__(
        self,
        *,
        name: str,
        data_root: Optional[str] = None,
        anno_file: Optional[str] = "data.json",
        csv_path: Optional[str] = None,
        features_path: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        max_words: int = 30,
        feature_framerate: float = 1.0,
        max_frames: int = 100,
        image_resolution: int = 224,
        debug: bool = False,
        **kwargs,
    ) -> None:
        # Get data root and load annotations
        self.data_root = get_data_root(data_root=data_root, config=None, cache_dir=FLAGEVALMM_DATASETS_CACHE_DIR)
        self.name = name
        self.csv_path = csv_path
        self.features_path = features_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.feature_framerate = feature_framerate
        self.image_resolution = image_resolution

        # Load annotations (captions)
        with open(osp.join(self.data_root, anno_file), 'r') as f:
            self.annotations = json.load(f)

        # Load the video captions from the csv
        self.data = pd.read_csv(csv_path)

        if debug:
            self.annotations = self.annotations[:160]  # For debugging, limit dataset size

        # Flatten the list of captions for easy access
        self.captions = [
            caption
            for annotation in self.annotations
            for caption in annotation["caption"][:5]  # Assume max 5 captions per image
        ]

        # Frame extractor for video frames
        self.frameExtractor = RawFrameExtractor(framerate=feature_framerate, size=image_resolution)

        # Special tokens for text processing
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self) -> int:
        """Returns the number of items in the dataset"""
        return len(self.data)

    def _get_text(self, caption: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tokenizes the caption and returns token IDs, masks, and segment IDs"""
        words = self.tokenizer.tokenize(caption)

        # Add CLS token and truncate to max_words
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_cls = self.max_words - 1
        if len(words) > total_length_with_cls:
            words = words[:total_length_with_cls]
        
        # Add SEP token
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        # Convert words to token IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        # Mask for the input (1 for valid token, 0 for padding)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # Padding to max_words length
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        return np.array(input_ids), np.array(input_mask), np.array(segment_ids)

    def _get_rawvideo(self, video_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract frames from the video with the specified ID"""
        video_mask = np.zeros((1, self.max_frames), dtype=np.long)
        video = np.zeros((1, self.max_frames, 1, 3, self.frameExtractor.size, self.frameExtractor.size), dtype=np.float)

        # Path to the video frames
        video_path = osp.join(self.features_path, video_id)

        # Get the raw video data
        raw_video_data = self.frameExtractor.get_video_data(video_path, self.max_frames)
        raw_video_data = raw_video_data['video']

        if len(raw_video_data.shape) > 3:
            raw_video_slice = self.frameExtractor.process_raw_data(raw_video_data)

            # If video length exceeds max_frames, sample the frames
            if self.max_frames < raw_video_slice.shape[0]:
                sample_idx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                video_slice = raw_video_slice[sample_idx, ...]
            else:
                video_slice = raw_video_slice

            # Fill the video data and mask
            slice_len = video_slice.shape[0]
            video_mask[0][:slice_len] = [1] * slice_len
            video[0][:slice_len, ...] = video_slice
        else:
            print("Error: Invalid video data")

        return video, video_mask

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns a single sample (text and video) given the index"""
        video_id = self.data['video_id'].values[idx]
        sentence = self.data['sentence'].values[idx]

        # Get the tokenized text data
        pairs_text, pairs_mask, pairs_segment = self._get_text(sentence)

        # Get the sampled video frames
        video, video_mask = self._get_rawvideo(video_id)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask

    def get_image(self, image_index: int) -> Dict[str, str]:
        """Returns the image path for a given image index"""
        annotation = self.annotations[image_index]
        return {"img_path": osp.join(self.data_root, annotation["img_path"])}

    def get_caption(self, caption_index: int) -> Dict[str, str]:
        """Returns the caption for a given caption index"""
        caption = self.captions[caption_index]
        return {"caption": caption}

    def meta_info(self) -> Dict[str, Any]:
        """Returns meta information about the dataset"""
        return {
            "name": self.name,
            "image_number": len(self.annotations),
            "caption_number": len(self.captions),
            "type": "retrieval",
        }

    def get_data(self, index: int, data_type: str) -> Dict[str, str]:
        """Returns the data for a given index and data type (img or text)"""
        if data_type == "img":
            return self.get_image(index)
        elif data_type == "text":
            return self.get_caption(index)
        else:
            raise Exception("Invalid data type")
