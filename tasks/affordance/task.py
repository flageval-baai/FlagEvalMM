config = dict(
    dataset_path="/share/project/benchmarks/ShareRobot",
    split="test",
    processed_dataset_path="affordance",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate", 
        pre_prompt="", 
        post_prompt="Only return the box in format: [x1, y1, x2, y2]." 
    ),
    name="affordance",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
