register_dataset = {"retrieval_msrvtt_dataset.py": "RetrievalMSRVTTDataset"}
register_evaluator = {"video_retrieval_evaluator.py": "RetrievalMSRVTTEvaluator"}

config = dict(
    dataset_path="shiyili1111/MSR-VTT",
    split="",
    processed_dataset_path="retrival/msrvtt/MSRVTT_JSFUSION",
    processor="msrvtt_process.py",
)

dataset = dict(
    type="RetrievalMSRVTTDataset",
    config=config,
    name="MSRVTT_JSFUSION_test",
)

evaluator = dict(
    type="RetrievalMSRVTTEvaluator",
)
