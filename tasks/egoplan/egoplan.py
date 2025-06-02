config = dict(
    dataset_path="ch-chenyu/All-Angles-Bench",
    split="train",
    processed_dataset_path="All-Angles-Bench",
    processor="process.py",
)


dataset = dict(
    type="VqaBaseDataset",
    data_root="/share/project/hezheqi/projects/OCRLiteData/ocrlite_v1",
    anno_file="data.json",
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    name="all_angles_bench",
)

evaluator = dict(type="BaseEvaluator")
