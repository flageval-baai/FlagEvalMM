from PIL import Image
import numpy as np
import av
import torch
from typing import Union
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)


def read_video_pyav(video_path: str, max_num_frames: int, return_tensors: bool = False):
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames

    # Handle videos without frame count metadata (e.g., FLV files)
    if total_frames == 0:
        # Try to estimate from duration and fps
        if container.duration and stream.average_rate:
            duration_sec = container.duration / av.time_base
            fps = float(stream.average_rate)
            total_frames = int(duration_sec * fps)

    # If still unknown, decode all frames first
    if total_frames == 0:
        all_frames = [frame for frame in container.decode(video=0)]
        total_frames = len(all_frames)
        if total_frames == 0:
            raise ValueError(f"No video frames found in {video_path}")
        if total_frames > max_num_frames:
            indices = np.arange(0, total_frames, total_frames / max_num_frames).astype(int)
        else:
            indices = np.arange(total_frames)
        frames = [all_frames[i] for i in indices]
    else:
        # Normal path: frame count is known
        if total_frames > max_num_frames:
            indices = np.arange(0, total_frames, total_frames / max_num_frames).astype(int)
        else:
            indices = np.arange(total_frames)
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        indices_set = set(indices)
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices_set:
                frames.append(frame)

    if return_tensors:
        tensors = (
            torch.from_numpy(np.stack([x.to_ndarray(format="rgb24") for x in frames]))
            .float()
            .cuda()
        )
        tensors = tensors.permute(0, 3, 1, 2)
        return tensors
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def load_image_or_video(
    image_or_video_path: str, max_num_frames: int, return_tensors: bool
) -> Union[np.ndarray, torch.Tensor]:
    logger.info(f"Loading image or video from {image_or_video_path}")
    if image_or_video_path.endswith(".png"):
        frame = Image.open(image_or_video_path)
        frame = frame.convert("RGB")
        frame = np.array(frame).astype(np.uint8)
        frame_list = [frame]
        frames = np.array(frame_list)
    elif image_or_video_path.endswith(".mp4"):
        # Use PyAV-based reader to avoid decord dependency
        frames = read_video_pyav(
            video_path=image_or_video_path,
            max_num_frames=max_num_frames,
            return_tensors=False,
        )
    else:
        frames = None

    if return_tensors:
        frames = torch.from_numpy(frames).float().cuda()
        frames = frames.permute(0, 3, 1, 2)

    return frames
