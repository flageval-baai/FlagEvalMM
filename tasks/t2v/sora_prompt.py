task_name = "t2v"

dataset = dict(
    type="Text2VideoBaseDataset",
    data_root="/share/project/mmdataset/sora_prompts_reformat.json",
    name="sora_prompt",
)

evaluator = dict(
    type="VideoScoreEvaluator", model="TIGER-Lab/VideoScore-v1.1", max_num_frames=48
)
