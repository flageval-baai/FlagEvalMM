task_name = "t2v_generation"

config = dict(
    dataset_path="fierytrees/UCF",
    processed_dataset_path="t2v/ucf",
    processor="process.py",
    split="test",
)


dataset = dict(
    type="Text2VideoBaseDataset",
    config=config,
    name="ucf-101",
)

clipsim_evaluator = dict(
    type="CLIPScoreEvaluator",
    model_name_or_path="openai/clip-vit-large-patch14",
    max_num_frames=48,
    start_method="spawn",
)

fvd_evaluator = dict(
    type="FVDEvaluator",
    config=config,
    model_path="flateon/FVD-I3D-torchscript",
    max_num_frames=48,
    start_method="spawn",
)

evaluator = dict(
    type="AggregationEvaluator",
    evaluators=[fvd_evaluator, clipsim_evaluator],
    start_method="spawn",
)
