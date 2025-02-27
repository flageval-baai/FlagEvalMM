config = dict(
    dataset_path="FlagEval/coco_val2014_sampled",
    split="test",
    processed_dataset_path="t2i/coco_1k_test",
    processor="process.py",
)

dataset = dict(
    type="Text2ImageBaseDataset",
    config=config,
    name="coco_1k_test",
)

clip_evaluator = dict(
    type="CLIPScoreEvaluator",
    model_name_or_path="openai/clip-vit-base-patch16",
)

inception_metric_evaluator = dict(
    type="InceptionMetricsEvaluator",
    metrics=["IS", "FID"],
    config=config,
)

vqascore_evaluator = dict(
    type="VqascoreEvaluator",
    model="clip-flant5-xxl",
)

evaluator = dict(
    type="AggregationEvaluator",
    evaluators=[clip_evaluator, inception_metric_evaluator, vqascore_evaluator],
)
