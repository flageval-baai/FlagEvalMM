config = dict(
    dataset_path="chanhee-luke/RoboSpatial-Home",
    split="all",
    processed_dataset_path="RoboSpatial-Home",
    processor="process.py",
)

post_prompt_point = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.

Each point should be represented as a normalized (x, y) tuple, where both x and y values are floats between 0 and 1000, corresponding to the position within the image (e.g., for a point at pixel (50, 75) in a 100*100 image, the normalized coordinate is (500, 750)).

Format your final answer strictly as follows on the last line of your response:
Answer: [(x1, y1), (x2, y2), ..., (xn, yn)]

Do not include additional text after this line.
"""

post_prompt_yes_no = """Your task is to answer the question above. Respond with a brief explanation if needed, followed by a yes or no answer in the last line of your response.

Format your final answer strictly as follows on the last line of your response:
Answer: yes or no

Do not include additional text after this line.
"""


def post_prompt(annotation: dict, **kwargs):
    question_type = annotation.get("question_type", "")
    if question_type == "point":
        return post_prompt_point
    else:
        return post_prompt_yes_no


dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt,
    ),
    config=config,
    name="robo_spatial_home_all_norm1k",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate_norm1k.py")
