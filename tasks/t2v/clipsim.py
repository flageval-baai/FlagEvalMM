task_name = "t2v"

dataset = dict(
    type="Text2VideoBaseDataset",
    data_root="/home/dengzijun/damo_video/sora_prompts_reformat.json",
    name="sora_prompt",
)

evaluator = dict(
    type="CLIPSIMEvaluator", model="openai/clip-vit-large-patch14", max_num_frames=48, start_method="spawn"
)
