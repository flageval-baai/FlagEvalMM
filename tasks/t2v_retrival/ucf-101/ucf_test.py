task_name = "t2v_retrieval"

config = dict(
    dataset_path="fierytrees/UCF",
    processed_dataset_path="t2v_retrieval/ucf",
    processor="process.py",
    split="test"
)


dataset = dict(
    type="VideoRetrievalDataset",
    config=config,
    name="UCF-101",
)

evaluator = dict(type="RetrievalEvaluator")