config = dict(
    dataset_path="/home/vlm/benchmarks/RoboVQA",
    split="all",
    processed_dataset_path="robovqa",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate", 
        pre_prompt="Answer the question using a single word or phrase .Question: ",   # prompt 前缀
        post_prompt=""   # prompt 后缀
    ),
    name="robovqa_all",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")


