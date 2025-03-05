
register_dataset = {"retrieval_msrvtt_dataset.py": "retrievalMsrvttDataset"}
register_evaluator = {"video_retrieval_evaluator.py": "videoRetrievalEvaluator"}

config = dict(
    dataset_path="BAAI/retrieval/msrvtt",
    split="test",
    processed_dataset_path="retrival/msrvtt",
    processor="../msrvtt/msrvtt_process.py",
)

dataset = dict(
    type="retrivalMsrvttDataset",
    config=config,
    name="msrvtt",
)

evaluator = dict(
    type="CmmuEvaluator",
)
