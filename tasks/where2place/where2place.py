config = dict(
    dataset_path="FlagEval/Where2Place",
    split="test",
    processed_dataset_path="Where2Place",
    processor="process.py",
)

post_prompt = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.

Each point should be represented as a normalized (x, y) tuple, where both x and y values are floats between 0 and 1, corresponding to the position within the image (e.g., for a point at pixel (50, 75) in a 100*100 image, the normalized coordinate is (0.5, 0.75)).

Format your final answer strictly as follows on the last line of your response:
Answer: [(x1, y1), (x2, y2), ..., (xn, yn)]

Do not include additional text after this line.
"""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt,
    ),
    config=config,
    name="Where2Place",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
