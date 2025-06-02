dataset = dict(
    type="VqaBaseDataset",
    data_root="/share/project/mmdataset/All-Angles-Bench",
    anno_file="data.json",
    prompt_template=dict(type="PromptTemplate", post_prompt="Answer with the option's letter from the given choices directly."),
    name="all_angles_bench",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["category"])
