task_name = "tv_retrieval"

config = dict(
    dataset_path="fierytrees/flag-eval-data",
    processed_dataset_path="tv_retrieval/ucf",
    processor="process.py",
    split="test"
)


dataset = dict(
    type="VideoRetrievalDataset",
    config=config,
    name="UCF-101",
)

evaluator = dict(type="RetrievalEvaluator")