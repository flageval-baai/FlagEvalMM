import json
import os
import os.path as osp
import pandas as pd
import shutil
from pathlib import Path
import subprocess


def process(cfg):
    """
    Process the MSRVTT dataset for video retrieval tasks.
    """
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    print(f"output_dir:{output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset
    download_dataset(data_dir, output_dir)
    
    # Process CSV file
    csv_path = os.path.join(output_dir, "MSRVTT_JSFUSION_test.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Create video directory structure
    video_dir = os.path.join(output_dir, "video")
    os.makedirs(video_dir, exist_ok=True)
    
    # Process data
    df = pd.read_csv(csv_path)
    content = []
    
    for _, row in df.iterrows():
        video_id = row['video_id']
        caption = row['sentence']
        
        # Create class directory
        class_name = video_id  # Using video_id as class_name
        class_dir = os.path.join(video_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy or link video file
        src_video = os.path.join(output_dir, "MSRVTT_Videos", f"{video_id}.mp4")
        dst_video = os.path.join(class_dir, f"{video_id}.mp4")
        
        if os.path.exists(src_video) and not os.path.exists(dst_video):
            try:
                # Create symbolic link
                os.symlink(os.path.abspath(src_video), dst_video)
            except OSError:
                # If symlink fails, copy the file
                shutil.copy2(src_video, dst_video)
        
        # Add to content
        info = {
            "prompt": caption,
            "id": video_id,
            "class_name": class_name
        }
        content.append(info)
    
    # Save data.json
    json.dump(content, open(osp.join(output_dir, "data.json"), "w"), indent=2)
    print(f"Processed {len(content)} entries. Data saved to {output_dir}/data.json")


def download_dataset(repo_id, output_dir):
    """
    Download the MSRVTT dataset from Hugging Face.
    """
    # Download dataset files
    command = (
        f"huggingface-cli download --repo-type dataset {repo_id} "
        f"--local-dir {output_dir} --resume-download"
    )
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Dataset downloaded successfully from {repo_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
