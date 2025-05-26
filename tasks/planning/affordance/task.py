config = dict(
    dataset_path="/home/vlm/finetune_json",
    split="test",
    processed_dataset_path="affordance",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate", 
        pre_prompt="",   # prompt 前缀
        post_prompt="Only return the box in format: [x1, y1, x2, y2]."   # prompt 后缀
    ),
    name="affordance",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
