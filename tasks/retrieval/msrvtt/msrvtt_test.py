task_name = "t2v_retrieval"

config = dict(
    dataset_path="shiyili1111/MSR-VTT",
    split="test",
    processed_dataset_path="retrieval/msrvtt",
    processor="msrvtt_process.py",
)

dataset = dict(
    type="VideoRetrievalDataset",
    config=config,
    name="MSRVTT_JSFUSION_test",
)

evaluator = dict(type="RetrievalEvaluator")
