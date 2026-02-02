task_name = "t2i"

config = dict(
    dataset_path="fierytrees/RelScene",
    split="test",
    processed_dataset_path="t2i/RelScene",
    processor="process.py",
)

dataset = dict(
    type="Text2ImageBaseDataset",
    config=config,
    name="RelScene",
)

clip_evaluator = dict(
    type="CLIPScoreEvaluator",
    model_name_or_path="openai/clip-vit-base-patch16",
    start_method="spawn",
)

inception_metric_evaluator = dict(
    type="InceptionMetricsEvaluator",
    metrics=["IS"],
    config=config,
    start_method="spawn",
)


evaluator = dict(
    type="AggregationEvaluator",
    evaluators=[clip_evaluator],
    start_method="spawn",
)
