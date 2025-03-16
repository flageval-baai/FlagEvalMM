register_dataset = {"video_dataset.py": "VideoDataset"}
register_evaluator = {"mvbench_dataset_evaluator.py": "MVBenchbenchDatasetEvaluator"}

config = dict(
    dataset_path="OpenGVLab/MVBench",
    split="test",
    processed_dataset_path="MVBench",
    processor="mvbench_process.py",
)

dataset = dict(
    type="VideoDataset",
    config=config,
    anno_file="data.json",
    name="mvbench",
)

evaluator = dict(type="MVBenchDatasetEvaluator")