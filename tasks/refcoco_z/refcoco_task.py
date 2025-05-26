config = dict(
    dataset_path="/share/project/benchmarks/Ref-L4",
    split="all",
    processed_dataset_path="Ref-L4",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate", 
        pre_prompt="Please provide the bounding box coordinate of the region this sentence describes: ",   # prompt 前缀
        post_prompt=""   # prompt 后缀
    ),
    name="ref-l4",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
