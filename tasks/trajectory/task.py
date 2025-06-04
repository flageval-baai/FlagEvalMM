config = dict(
    dataset_path="/share/project/benchmarks/ShareRobot",
    split="test",
    processed_dataset_path="trajectory",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate", 
        pre_prompt="",   # prompt 前缀
        post_prompt=""   # prompt 后缀
    ),
    name="trajectory",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
