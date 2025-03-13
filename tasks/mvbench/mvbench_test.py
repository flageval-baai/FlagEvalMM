register_dataset = {"mvbench_dataset.py": "MVBenchDataset"}
register_evaluator = {"mvbench_dataset_evaluator.py": "MLUVDatasetEvaluator"}
dataset = dict(
    type="MVBenchDataset",
    data_root="OpenGVLab/MVBench",
    split="test",
    name="mvbench",
)

evaluator = dict(type="MLUVDatasetEvaluator")