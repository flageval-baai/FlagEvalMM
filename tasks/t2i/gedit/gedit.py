dataset = dict(
    type="Image2ImageBaseDataset",
    data_root="/share/project/mmdataset/t2i/GEdit-Bench",
    name="gedit",
    anno_file="data.json",
    source_key="input_image_raw",
    prompt_key="instruction",
    id_key="question_id",
)

gedit_evaluator = dict(
    type="GEditEvaluator",
    data_root="/share/project/mmdataset/t2i/GEdit-Bench",
    input_image_key="input_image_raw",
    fallback_input_key="input_image",
    instruction_key="instruction",
    language_key="instruction_language",
    task_type_key="task_type",
    intersection_key="Intersection_exist",
    model="gpt-4o-2024-05-13",
    url="https://api.pandalla.ai/v1/chat/completions",
    api_key="sk-97kkUEMSmVgcczfeA8O3hCUuairzRcezYAZb5A5cBoeHVQkD",
    language="en",
    max_workers=8,
    resize_area=512 * 512,
)


evaluator = dict(
    type="AggregationEvaluator",
    evaluators=[gedit_evaluator],
    start_method="spawn",
)
