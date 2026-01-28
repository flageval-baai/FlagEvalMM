robo_post_prompt_point = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.

Each point should be represented as a normalized (x, y) tuple, where both x and y values are floats between 0 and 1, corresponding to the position within the image (e.g., for a point at pixel (50, 75) in a 100*100 image, the normalized coordinate is (0.5, 0.75)).

Format your final answer strictly as follows on the last line of your response:
Answer: [(x1, y1), (x2, y2), ..., (xn, yn)]

Do not include additional text after this line.
"""

robo_post_prompt_yes_no = """Your task is to answer the question above. Respond with a brief explanation if needed, followed by a yes or no answer in the last line of your response.

Format your final answer strictly as follows on the last line of your response:
Answer: yes or no

Do not include additional text after this line.
"""

where2place_post_prompt = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.

Each point should be represented as a normalized (x, y) tuple, where both x and y values are floats between 0 and 1, corresponding to the position within the image (e.g., for a point at pixel (50, 75) in a 100*100 image, the normalized coordinate is (0.5, 0.75)).

Format your final answer strictly as follows on the last line of your response:
Answer: [(x1, y1), (x2, y2), ..., (xn, yn)]

Do not include additional text after this line.
"""

common_post_prompt = "Carefully analyze the multiple-choice question above and reason through it step by step. Conclude your response with a line in the following format: Answer: $LETTER (without quotes), where $LETTER is the letter of the correct choice."
common_post_prompt2 = "Answer with the option's letter from the given choices directly."

PROMPT_MAP = {
    "SAT": common_post_prompt,
    "erqa": common_post_prompt,
    "Where2Place": where2place_post_prompt,
    "all_angles_bench": common_post_prompt2,
    "egoplan_bench2": common_post_prompt2,
    "cv_bench_test": common_post_prompt2,
    "embspatial_bench": common_post_prompt2,
    "blink_val_ev": "The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.",
    "vsi_bench_tiny": {
        "pre_prompt": "These are frames of a video.",
        "post_prompt": {
            "multiple-choice": "Carefully analyze the question above and reason through it step by step. Conclude your response with a line in the following format:\nAnswer: $LETTER (without quotes), where $LETTER corresponds to the correct option.",
            "numerical": "Carefully analyze the question above and reason through it step by step. Conclude your response with a line in the following format:\nAnswer: $NUMBER (without quotes), where $NUMBER is a number (integer or float) corresponds to the correct answer.",
        },
    },
    "robo_spatial_home_all": {
        "post_prompt": {
            "yes-no": robo_post_prompt_yes_no,
            "point": robo_post_prompt_point,
        }
    },
}
