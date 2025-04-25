task_name = "t2v"

dataset = dict(
    type="Text2VideoBaseDataset",
    data_root="/home/dengzijun/damo_video/ucf_prompts_reformat.json",
    name="ucf-101",
)

clipsim_evaluator = dict(
    type="CLIPSIMEvaluator", model="openai/clip-vit-large-patch14", max_num_frames=48, start_method="spawn"
)

fvd_evaluator = dict(
    type="FVDEvaluator", example_dir="/home/dengzijun/output/sora_prompt/", model_path="/home/dengzijun/damo_video/i3d_torchscript.pt", max_num_frames=48, start_method="spawn"
)

evaluator = dict(
    type="AggregationEvaluator",
    evaluators=[clipsim_evaluator, fvd_evaluator],
    start_method="spawn"
)