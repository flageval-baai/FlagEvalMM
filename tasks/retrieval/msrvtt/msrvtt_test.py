config = dict(
    dataset_path="nlphuji/msrvtt_video_text_retrieval",
    split="test",
    processed_dataset_path="retrieval/msrvtt",
    processor="../msrvtt/msrvtt_process.py",
)

dataset = dict(type="MSRVTT_single_sentence_dataLoader", config=config, name="msrvtt")

evaluator = dict(type="VideoRetrievalEvaluator")
