config = dict(
    dataset_path="https://huggingface.co/datasets/BaiqiL/GenAI-Bench-1600/raw/main/genai_image.json",
    split="test_1600",
    processed_dataset_path="t2i/genai_bench",
    processor="process.py",
)

dataset = dict(
    type="Text2ImageBaseDataset",
    config=config,
    name="genai_bench_test1600",
)

vqascore_evaluator = dict(
    type="VqascoreEvaluator",
    model="clip-flant5-xxl",
    start_method="spawn",
)

one_align_evaluator = dict(
    type="OneAlignEvaluator",
    model="q-future/one-align",
    metrics=["quality", "aesthetics"],
)

evaluator = dict(
    type="AggregationEvaluator",
    evaluators=[one_align_evaluator, vqascore_evaluator],
    start_method="spawn",
)
