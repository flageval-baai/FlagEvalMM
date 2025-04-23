config = dict(
    dataset_path="FlagEval/sub_spatial",
    split="test",
    processed_dataset_path="sub_spatial",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    config=config,
    name="sub_spatial",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
