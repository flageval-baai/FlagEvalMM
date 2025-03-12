import json
import os
import os.path as osp
from datasets import load_dataset
from PIL import Image
from download import download_and_extract_dataset
# from dataloaders.rawframe_util import RawFrameExtractor
def process(cfg):
    """
    处理 MSRVTT 数据集，将视频帧保存并创建对应的 JSON 文件
    """

    download_and_extract_dataset(cfg.dataset_path,"./",cfg.extract_dir)
    
download_and_extract_dataset("shiyili1111/MSR-VTT", "./", "./")    