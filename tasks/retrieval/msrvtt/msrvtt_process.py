import json
import os
import os.path as osp
from datasets import load_dataset
from PIL import Image
from dataloaders.rawframe_util import RawFrameExtractor

def process(cfg):
    """
    处理 MSRVTT 数据集，将视频帧保存并创建对应的 JSON 文件
    """
    data_dir, split = cfg.dataset_path, cfg.split

    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    content = []
    
    # 加载 MSRVTT 数据集
    data = load_dataset(data_dir, name=name, split=split)
    
    # 创建输出目录
    os.makedirs(osp.join(output_dir, "image"), exist_ok=True)
    
    # 设置视频帧提取器
    frame_extractor = RawFrameExtractor(framerate=1.0, size=224)  # 假设我们从视频中提取224x224的帧

    # 遍历每个视频及其字幕
    for i, annotation in enumerate(data):
        video_id = annotation["video_id"]
        captions = annotation["caption"]
        
        # 初始化保存该视频信息的字典
        video_info = {
            "video_id": video_id,
            "captions": captions,
            "img_paths": []
        }

        # 提取视频帧
        video_path = osp.join(data_dir, video_id)  # 假设视频在数据集中按 video_id 命名
        video_frames = frame_extractor.get_video_data(video_path, max_frames=100)["video"]  # 获取最多100帧的视频数据

        # 保存帧图像并记录路径
        for j, frame in enumerate(video_frames):
            frame_path = osp.join(output_dir, "image", f"{video_id}_frame_{j}.jpg")
            Image.fromarray(frame).save(frame_path)
            video_info["img_paths"].append(f"image/{video_id}_frame_{j}.jpg")

        # 将视频信息添加到 content 列表
        content.append(video_info)

    # 保存数据为 JSON 文件
    json.dump(content, open(osp.join(output_dir, "data.json"), "w"), indent=2)
