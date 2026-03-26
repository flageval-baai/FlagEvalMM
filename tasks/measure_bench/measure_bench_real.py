register_evaluator = {"measure_bench_evaluator.py": "MeasureBenchEvaluator"}

_base_ = "post_prompt.py"

config = dict(
    dataset_path="FlagEval/MeasureBench",
    split="real_world",
    processed_dataset_path="MeasureBench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    name="measure_bench_real",
)

evaluator = dict(
    type="MeasureBenchEvaluator",
    tracker_type="image_type",
    tracker_subtype="design",
)
