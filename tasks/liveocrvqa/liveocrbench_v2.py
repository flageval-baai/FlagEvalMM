config = dict(
    dataset_path="BAAI/LiveOCRVQA",
    split='image',
    data_file="liveocr_v2.json",
    processed_dataset_path="LiveOCRBench_addition_final_v2",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(type="PromptTemplate", pre_prompt="", post_prompt=""),
    name="liveocr_bench_final_v2",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
