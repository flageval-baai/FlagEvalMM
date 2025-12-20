dataset = dict(
    type="Text2ImageBaseDataset",
    data_root="/share/project/mmdataset/t2i/LongText-Bench",
    name="longtext_bench",
)

base_url = "http://172.27.72.180:8000/v1/chat/completions"
api_key = "EMPTY"

longtext_bench_evaluator = dict(
    type="LongTextBenchEvaluator",
    model="Qwen2.5-VL-7B-Instruct",
    base_url=base_url,
    api_key=api_key,
    max_workers=10,
    temperature=0.0,
    max_tokens=1024,
    max_image_size=None,
    min_short_side=None,
    max_long_side=None,
)


evaluator = dict(
    type="AggregationEvaluator",
    evaluators=[longtext_bench_evaluator],
    start_method="spawn",
)
