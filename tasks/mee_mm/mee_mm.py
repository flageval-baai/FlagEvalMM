# 任务基础配置
config = dict(
    # 通用考试数据集路径，您可以替换为实际的Hugging Face仓库ID
    dataset_path="Han03430/mee_mm",
    split="test",  # or "val", "train"
    # 处理后的数据集将保存在 ~/.cache/flagevalmm/mee_mm_bench/
    processed_dataset_path="mee_mm_bench",
    # 数据处理脚本
    processor="process.py",
)

# 数据集定义
dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate",
        pre_prompt="请根据提供的题目信息以及参考图片回答问题。",
        post_prompt="请给出您的答案。",
    ),
    name="mee_mm_bench",
)

# 评估器定义
evaluator = dict(
    type="BaseEvaluator",
    # 评估逻辑脚本
    eval_func="evaluate.py"
)
