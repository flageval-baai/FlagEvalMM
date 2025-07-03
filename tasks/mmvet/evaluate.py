import json
import time
from typing import Dict, List, Tuple
import pandas as pd
from collections import Counter
from flagevalmm.models import GPT
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)

MM_VET_PROMPT = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <image i> in the question indicates where an i-th image is. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right).
Give you a table example:
| Question | Ground truth | Prediction | Correctness |
| --- | --- | --- | --- |
| What is x in the equation?<image 1> | -1 <AND> -5 | x = 3 | 0.0 |
| What is x in the equation?<image 1> | -1 <AND> -5 | x = -1 | 0.5 |
| What is x in the equation?<image 1> | -1 <AND> -5 | x = -5 | 0.5 |
| What is x in the equation?<image 1> | -1 <AND> -5 | x = -5 or 5 | 0.5 |
| What is x in the equation?<image 1> | -1 <AND> -5 | x = -1 or x = -5 | 1.0 |
| Can you explain this meme?<image 1> | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4 |
| Can you explain this meme?<image 1> | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0 |
You need to predict the correctness of the following data:
{DATA}
Return the correctness score (0.0-1.0) for the prediction as json in the following format:
{{"score": correctness_score}}
"""


def get_meta_info(annotations: Dict) -> List[pd.DataFrame]:
    cap_counter = Counter()
    cap_detail_counter = Counter()
    for _, anno in annotations.items():
        cap = set(anno["capability"])
        cap_counter.update(cap)
        cap_detail_counter.update(["_".join(list(cap))])
    print(cap_counter, cap_detail_counter)
    cap_columns = pd.DataFrame(cap_counter.keys())
    cap_details_columns = pd.DataFrame(cap_detail_counter.keys())
    return cap_columns, cap_details_columns


def get_score(annotations: Dict, pred: str, model: GPT) -> Tuple[float, bool]:
    question = annotations["question"]
    question_id = annotations["question_id"]
    answer = annotations["answer"]

    gpt_query_prompt = "| " + " | ".join(
        [
            question.replace("\n", "<br>"),
            answer.replace("<AND>", " <AND> ")
            .replace("<OR>", " <OR> ")
            .replace("\n", "<br>"),
            pred.replace("\n", "<br>"),
            "",
        ]
    )

    gpt_query_prompt = MM_VET_PROMPT.format(DATA=gpt_query_prompt)

    message = model.build_message(query=gpt_query_prompt)
    grade_sample_run_complete = False
    val_success = False
    temperature = 0.0
    while not grade_sample_run_complete:
        response = model.infer(
            chat_messages=message, temperature=temperature, max_tokens=16
        )

        if response:
            try:
                data = json.loads(response.content)
                score = data["score"]
                if 0.0 <= score <= 1.0:
                    grade_sample_run_complete = True
                    val_success = True
            except Exception:
                time.sleep(1)
                temperature += 0.5
                logger.info(
                    f"Sleep 1 secs, {question_id} try again with increased temperature {temperature}."
                )
                if temperature >= 2:  # Assuming a max temperature threshold
                    score = 0.0
                    grade_sample_run_complete = True
                    logger.info(
                        f"Reach to max trials, {question_id} failed to get a score."
                    )
        else:
            score = 0.0
            grade_sample_run_complete = True
            logger.info(f"{question_id} failed to get a score.")
    return score, val_success


def get_result(annotations: Dict, predictions: List[Dict], llm_evaluator: GPT) -> Dict:
    results = {}
    cap_columns, cap_details_columns = get_meta_info(annotations)
    overall_score = 0
    for pred in predictions:
        anno = annotations[pred["question_id"]]
        score, val_success = get_score(anno, pred["answer"], llm_evaluator)
        pred["score"] = score
        pred["capability"] = anno["capability"]
        pred["answer"] = anno["answer"]
        pred["val_success"] = val_success
        overall_score += score
    overall_score = round(overall_score / len(predictions) * 100, 4)
    cap_scores = {cap: 0 for cap in cap_columns.squeeze().tolist()}
    cap_details_scores = {
        detail: 0 for detail in cap_details_columns.squeeze().tolist()
    }

    # Count the number of results for each capability and detail
    cap_counts = {cap: 0 for cap in cap_scores}
    cap_details_counts = {detail: 0 for detail in cap_details_scores}

    # Aggregate scores for each capability and detail
    for result in predictions:
        for cap in cap_scores:
            if cap in result["capability"]:
                cap_scores[cap] += result["score"]
                cap_counts[cap] += 1
        for detail in cap_details_scores:
            detail_set = set(detail.split("_"))
            result_detail_set = set(result["capability"])
            if detail_set == result_detail_set:
                cap_details_scores[detail] += result["score"]
                cap_details_counts[detail] += 1

    # Calculate the average score for each capability
    for cap in cap_scores:
        if cap_counts[cap] > 0:
            cap_scores[cap] = cap_scores[cap] / cap_counts[cap] * 100
        logger.info(f"Score for {cap}: {cap_scores[cap]:.2f}")

    # Calculate the average score for each detailed capability
    for detail in cap_details_scores:
        if cap_details_counts[detail] > 0:
            cap_details_scores[detail] = (
                cap_details_scores[detail] / cap_details_counts[detail] * 100
            )
        logger.info(f"Score for {detail}: {cap_details_scores[detail]:.2f}")
    results.update(
        {
            "accuracy": overall_score,
            "capability_scores": cap_scores,
            "capability_detail_scores": cap_details_scores,
        }
    )
    return results
