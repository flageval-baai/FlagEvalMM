register_evaluator = {"rome_evaluator.py": "ROMEEvaluator"}

_base_ = "post_prompt.py"

dataset = dict(
    type="VqaBaseDataset",
    data_root="/share/project/hezheqi/data/ROME-V/ROME-I",
    anno_file="ROME-I-memes.json",
    name="rome_i_memes",
)

evaluator = dict(type="ROMEEvaluator", tracker_type="image_type", tracker_subtype="image_subtype")