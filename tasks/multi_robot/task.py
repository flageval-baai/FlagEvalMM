config = dict(
    dataset_path="/share/project/benchmarks/multi-robot",
    split="test",
    processed_dataset_path="multi_robot",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate", 
        pre_prompt="",  
        post_prompt=""   
    ),
    name="multi_robot",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
