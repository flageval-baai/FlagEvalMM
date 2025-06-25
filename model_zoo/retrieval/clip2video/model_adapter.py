import os
import cv2
from typing import Dict, Any
from transformers import CLIPModel
from transformers import CLIPProcessor
import torch
import numpy as np
from PIL import Image
from flagevalmm.models.base_model_adapter import BaseModelAdapter

from flagevalmm.server.utils import parse_args, get_retrieval_data


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        ckpt_path = task_info["model_path"]
        self.model = CLIPModel.from_pretrained(ckpt_path).cuda()
        self.preprocess = CLIPProcessor.from_pretrained(ckpt_path)

    def get_video(self, video_id, task_name):
        response = get_retrieval_data(
            index=video_id,
            task_name=task_name,
            data_type="video",
            server_ip=self.server_ip,
            server_port=self.server_port,
            timeout=self.timeout,
        )
        video_path = response["video_path"]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        video_features = []
        for frame in frames:
            frame_image = Image.fromarray(frame).convert("RGB")
            processed_frame = self.preprocess(images=frame_image, return_tensors="pt")
            video_features.append(processed_frame["pixel_values"].squeeze(0))
        video_features = torch.stack(video_features).mean(dim=0)
        return video_features.squeeze(0).cuda()

    def get_caption(self, caption_id, task_name):
        response = get_retrieval_data(
            index=caption_id,
            task_name=task_name,
            data_type="text",
            server_ip=self.server_ip,
            server_port=self.server_port,
            timeout=self.timeout,
        )
        return response["caption"]

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        output_dir = meta_info["output_dir"]
        max_cap_len = 77
        itd = 50
        N = meta_info["video_number"]

        acc_video_embeddings = None
        acc_text_embeddings = None

        for i in range(0, N, itd):
            if i % itd == 0:
                print("{}/{}=={}%".format(i, N, 100.0 * i / N))
            _s, _e = i, min(i + itd, N)

            videos = [self.get_video(video_id, task_name) for video_id in range(_s, _e)]
            videos = torch.stack(videos, 0).squeeze()

            captions = [
                self.get_caption(caption_id, task_name) for caption_id in range(_s, _e)
            ]
            texts = [
                self.preprocess(
                    text=cap,
                    padding="max_length",
                    truncation=True,
                    max_length=max_cap_len,
                    return_tensors="pt",
                )["input_ids"].cuda()
                for cap in captions
            ]
            texts = torch.cat(texts, 0)

            with torch.no_grad():
                video_features = self.model.get_image_features(videos)
                text_features = self.model.get_text_features(texts)
                video_embeddings = video_features / video_features.norm(
                    dim=-1, keepdim=True
                )
                text_embeddings = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )

            torch.cuda.empty_cache()

            # Convert to numpy and accumulate
            batch_video_embeddings = video_embeddings.cpu().numpy()
            batch_text_embeddings = text_embeddings.cpu().numpy()

            if acc_video_embeddings is None:
                acc_video_embeddings = batch_video_embeddings
                acc_text_embeddings = batch_text_embeddings
            else:
                acc_video_embeddings = np.concatenate(
                    (acc_video_embeddings, batch_video_embeddings), axis=0
                )
                acc_text_embeddings = np.concatenate(
                    (acc_text_embeddings, batch_text_embeddings), axis=0
                )

        # Final similarity computation
        acc_video_embeddings = torch.from_numpy(acc_video_embeddings).cuda()
        acc_text_embeddings = torch.from_numpy(acc_text_embeddings).cuda()
        acc_sim = acc_video_embeddings.mm(acc_text_embeddings.T)
        acc_sim = acc_sim.cpu().numpy()

        full_save_path = os.path.join(output_dir, meta_info["name"])
        np.save(f"{full_save_path}", acc_sim)

        return acc_sim


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        extra_cfg=args.cfg,
        local_mode=args.local_mode,
        task_names=args.tasks,
        output_dir=args.output_dir,
        model_path=args.model,
        debug=args.debug,
        quiet=args.quiet,
    )
    model_adapter.run()
