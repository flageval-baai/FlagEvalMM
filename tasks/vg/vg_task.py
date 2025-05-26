config = dict(
    dataset_path="/home/vlm/benchmarks/BAAI-VG",
    split="all",
    processed_dataset_path="baai-vg",
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
    name="baai-vg",
)

evaluator = dict(
    type="BaseEvaluator", 
    eval_func="evaluate.py"
)