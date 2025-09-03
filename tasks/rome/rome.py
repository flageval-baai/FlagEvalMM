register_evaluator = {"rome_evaluator.py": "ROMEEvaluator"}

_base_ = "post_prompt.py"

dataset = dict(
    type="VqaBaseDataset",
    data_root="BAAI/ROME/data",
    anno_file="test.json",
    name="rome",
)

evaluator = dict(
    type="ROMEEvaluator",
    tracker_type="image_type",
    tracker_subtype="image_subtype",
    aggregation_fields=["raw_answer"],
)
