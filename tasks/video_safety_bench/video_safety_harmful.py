_base_ = "video_safety_base.py"

config = dict(
    dataset_path="BAAI/Video-SafetyBench",
    processed_dataset_path="Video-SafetyBench",
    split="harmful",
    processor="process.py",
)

dataset = dict(
    config=config,
    name="video_safety_harmful",
)
