
register_dataset = {"retrieval_msrvtt_dataset.py": "retrievalMsrvttDataset"}
register_evaluator = {"video_retrieval_evaluator.py": "videoRetrievalEvaluator"}

config = dict(
    dataset_path="shiyili1111/MSR-VTT",
    split="test",
    processed_dataset_path="retrival/msrvtt/MSRVTT_JSFUSION_test",
    processor="msrvtt_process.py",
    extract_dir = "./",  
)

dataset = dict(
    type="retrivalMsrvttDataset",
    config=config,
    name="MSRVTT_JSFUSION_test",
)

evaluator = dict(
    type="videoRetrievalEvaluator",
)
