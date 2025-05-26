config = dict(
    dataset_path="/share/project/tangyingbo/dataset/where2place/",
    split="test",
    processed_dataset_path="Where2Place",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="Answer the question above. Think through the problem step by step before providing your final answer.",
    ),
    config=config,
    name="Where2Place",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
