# config = dict(
#     dataset_path="https://huggingface.co/datasets/BaiqiL/GenAI-Bench-1600/raw/main/genai_image.json",
#     split="test_1600",
#     processed_dataset_path="t2i/genai_bench",
#     processor="process.py",
# )

dataset = dict(
    type="Text2ImageBaseDataset",
    data_root="/share/project/mmdataset/t2i/wise",
    name="wise_1k",
)

base_url = "https://api.pandalla.ai/v1/chat/completions"
api_key = "sk-97kkUEMSmVgcczfeA8O3hCUuairzRcezYAZb5A5cBoeHVQkD"

wise_evaluator = dict(
    type="WiseEvaluator",
    model="gpt-4o-2024-05-13",
    base_url=base_url,
    api_key=api_key,
    max_workers=10,
    validate_prompt_ids=False,
    category="all",
)


evaluator = dict(
    type="AggregationEvaluator",
    evaluators=[wise_evaluator],
    start_method="spawn",
)
