register_dataset = {"retrieval_msrvtt_dataset.py": "RetrievalMSRVTTDataset"}

config = dict(
    dataset_path="shiyili1111/MSR-VTT",
    split="",
    processed_dataset_path="t2v_retrieval/msrvtt",
    processor="msrvtt_process.py",
)

dataset = dict(
    type="RetrievalMSRVTTDataset",
    config=config,
    name="MSRVTT_JSFUSION_test",
)

evaluator = dict(
    type="RetrievalEvaluator",
)
