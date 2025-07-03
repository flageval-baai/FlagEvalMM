import json
from copy import deepcopy
from typing import Dict, List, Tuple

from flagevalmm.models import GPT

DESCRIPTIVE_GRADING_PREFIX = """
You will be given <|NUM_TRIPLETS|> pairs of ground truth answers and model responses under an overarching question. You need to go through each of the pairs, extract the final answer from the model response, compare it with the ground truth answer, and then assign a binary score. Avoid providing explanations in your response. If there is no provided model response, please leave the extracted answer empty and give a score of 0. Your response must follow json formats with keys [<|JSON_KEYS|>] where the value for any `extract_answer` is your extracted answer and `score` is an interger in [0, 1] based on the following rules:\n

Overarching Question: <|OVERARCHING_QUESTION|>
"""

DESCRIPTIVE_GRADING_QMAP = {
    1: "What is the title of the plot?",
    2: "What is the label of the x-axis?",
    3: "What is the label of the y-axis?",
    4: "What is the leftmost labeled tick on the x-axis?",
    5: "What is the rightmost labeled tick on the x-axis?",
    6: "What is the spatially lowest labeled tick on the y-axis?",
    7: "What is the spatially highest labeled tick on the y-axis?",
    8: "What is difference between consecutive numerical tick values on the x-axis?",
    9: "What is difference between consecutive numerical tick values on the y-axis?",
    10: "How many lines are there?",
    11: "Do any lines intersect?",
    12: "How many discrete labels are there in the legend?",
    13: "What are the names of the labels in the legend? (from top to bottom, then left to right)",
    14: "What is the difference between the maximum and minimum values of the tick labels on the continuous legend (i.e., colorbar)?",
    15: "What is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?",
    16: "What is the general trend of data from left to right?",
    17: "What is the total number of explicitly labeled ticks across all axes?",
    18: "What is the layout of the subplots?",
    19: "What is the number of subplots?",
}

DESCRIPTIVE_GRADING_ICL = {
    "title": """
Rubric:
    * Give a score of 1 if and only if the extracted answer and the ground truth answer are referring to the same term. It's acceptable to have different grammar or form (e.g., α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). It's acceptable to omit letter prefixes (e.g., (a) Increment over time and Increment over time).
    * Give a score of 0 if any term in the extracted answer is different from the ground truth answer.
    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.

    ### Example Start ###
    T1:
    Response 1: The title of the plot is "The number of students in each grade".
    Ground Truth 1: The variance of students in each grade

    T2:
    Response 2: There is no title.
    Ground Truth 2: Not Applicable

    T3:
    Response 3: A_v^t
    Ground Truth 3: A^t_v

    {
        "extract_answer_T1": "The number of students in each grade",
        "score_T1": 0
        "extract_answer_T2: "Not Applicable",
        "score_T2": 1
        "extract_answer_T3": "A_v^t",
        "score_T3": 1
    }
    ### Example End ###
""",
    "ocr": """
Rubric:
    * Give a score of 1 if and only if the extracted answer and the ground truth answer are referring to the same term. It's acceptable to have equivalent grammar or form (e.g., α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). If the ground truth is a number, the extracted answer should be the number with the exact same value.
    * Give a score of 0 if any term in the extracted answer is different from the ground truth answer, or if the extracted number is different in value from the ground truth number.
    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.

    ### Example Start ###
    T1:
    Response 1: The answer is 1.0
    Ground Truth 1: 1.00

    T2:
    Response 2: By manually inspecting the plot, the final answer should be 0.
    Ground Truth 2: Not Applicable

    T3:
    Response 3: A_v^t
    Ground Truth 3: A^t_v

    {
        "extract_answer_T1": 1.0,
        "score_T1": 1
        "extract_answer_T2: 0,
        "score_T2": 0
        "extract_answer_T3": "A_v^t",
        "score_T3": 1
    }
    ### Example End ###
""",
    "quant": """
Rubric:
    * Give a score of 1 if and only if the extracted answer and the ground truth answer are numbers with the exact same value.
    * Give a score of 0 if the extracted answer is different in value from the ground truth answer.
    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.

    ### Example Start ###
    T1:
    Response 1: 5
    Ground Truth 1: 6

    T2:
    Response 2: 0
    Ground Truth 2: Not Applicable

    T3:
    Response 3: 4
    Ground Truth 3: 4

    {
        "extract_answer_T1": 5,
        "score_T1": 0
        "extract_answer_T2: 0,
        "score_T2": 0
        "extract_answer_T3": 4,
        "score_T3": 1
    }
    ### Example End ###
""",
    "bool": """
Rubric:
    * Give a score of 1 if and only if the extracted answer and the ground truth answer are the same.
    * Give a score of 0 if the extracted answer and the ground truth answer are different.
    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.

    ### Example Start ###
    T1:
    Response 1: No, there are no intersections.
    Ground Truth 1: no

    T2:
    Response 2: No, all the lines are parallel.
    Ground Truth 2: Yes

    T3:
    Response 3: There are no lines in the plot.
    Ground Truth 3: Not Applicable

    {
        "extract_answer_T1": "No",
        "score_T1": 1
        "extract_answer_T2: "No",
        "score_T2": 0
        "extract_answer_T3": "Not Applicable",
        "score_T3": 1
    }
    ### Example End ###
""",
    "enum": """
Rubric:
    * Give a score of 1 if and only if the extracted answer and the ground truth answer are referring to the same term. It's acceptable to have equivalent grammar or form (e.g., α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). The order of the terms must be the same.
    * Give a score of 0 if any term in the extracted answer is different from the ground truth answer, or if the order of the terms is different.
    * When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a score of 1.

    ### Example Start ###
    T1:
    Response 1: Here are the names of the labels: A, B, C
    Ground Truth 1: B, A, C

    T2:
    Response 2: The labels are T56, B33.
    Ground Truth 2: T56,B33,A12

    T3:
    Response 3: \alpha, \beta, \\gamma^t_v
    Ground Truth 3: α, β, γ_v^t

    {
        "extract_answer_T1": "A, B, C",
        "score_T1": 0
        "extract_answer_T2: "T56, B33",
        "score_T2": 0
        "extract_answer_T3": "\alpha, \beta, \\gamma^t_v",
        "score_T3": 1
    }
    ### Example End ###
""",
    "trend": """
Rubric:
    * Give a score of 1 if and only if the extracted answer and the ground truth answer share the same general trend.
    * Give a score of 0 if the extracted answer and the ground truth answer are different in trend expression.

    ### Example Start ###
    T1:
    Response 1: there is an increase in the data from left to right
    Ground Truth 1: Decreases

    T2:
    Response 2: the curves move up and stay constant
    Ground Truth 2: Increases then stabilizes

    T3:
    Response 3: Decreases
    Ground Truth 3: Decreases then increases

    {
        "extract_answer_T1": "Increases",
        "score_T1": 0
        "extract_answer_T2: "Move up and stay constant",
        "score_T2": 1
        "extract_answer_T3": "Decreases",
        "score_T3": 0
    }
    ### Example End ###
""",
    "layout": """
Rubric:
    * Give a score of 1 if and only if the extracted answer and the ground truth answer are the same in terms of the number of rows and columns (e.g., n by m).
    * Give a score of 0 if the extracted answer is different from the ground truth answer.

    ### Example Start ###
    T1:
    Response 1: 2 by 3
    Ground Truth 1: 3 by 2

    T2:
    Response 2: the layout is 1 by 1
    Ground Truth 2: 1 by 1

    T3:
    Response 3: there are two rows and three columns
    Ground Truth 3: 2 by 3

    {
        "extract_answer_T1": "2 by 3",
        "score_T1": 0
        "extract_answer_T2: "1 by 1",
        "score_T2": 1
        "extract_answer_T3": "2 by 3",
        "score_T3": 1
    }
    ### Example End ###
""",
}

REASONING_GRADING_PREFIX = """
You will be given a question, an ground truth answer and a model response. You need to extract the final answer from the model response, compare it with the ground truth answer, and then assign a binary score. Avoid providing explanations in your response. If there is no provided model response, please leave the extracted answer empty and give a score of 0.

Your response must follow json formats with keys [extract_answer, score] where the value of the score is an interger in [0, 1]. You must follow the scoring rules:\n"""

REASONING_GRADING_INST = {
    1: """
    ### Rules ###
    * Give a score of 1 if and only if the final answer and the ground truth answer are referring to the same term. It's acceptable to have different grammar or form (e.g., α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). It's also acceptable to have different orders of the terms when question asks for multiple terms.
    * Give a score of 0 if any term (e.g., ACC+ and ACC; P-101 and P=101) is different between the final answer and the ground truth.

    ### Example 1 Starts ###
    * Question: What is the name of the curve that intersects y=\\lambda exactly three times?
    * Ground Truth: P56962
    * Response: There is only one curve that intersects y=\\lambda exactly three times. The name of the curve is written as P55762.

    {
        "extracted_answer": "P55762",
        "score": 0
    }
    ### Example 1 Ends ###


    ### Example 2 Starts ###
    * Question: What is the letter of the subplot where all bars are above 35?
    * Ground Truth: (b)
    * Response: The letter of the subplot where all bars are above 35 is b.

    {
        "extracted_answer": "b",
        "score": 1
    }
    ### Example 2 Ends ###

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>

    """,
    2: """
    ### Rules ###
    * If there are predefined options in the question:
        * Give a score of 1 if the final answer matches the ground truth answer exactly.
        * Give a score of 0 if the final answer does not match the ground truth answer.
    * If there are no predefined options in the question:
        * Give a score of 1 if the final answer shares the same semantic meaning with the ground truth answer (e.g., "increasing then decreasing" and "moving up then down"; "converge" and "move closer together").
        * Give a score of 0 if the final answer shares different semantic meanings from the ground truth answer (e.g., "increasing then decreasing" and "remain constant"; "converge" and "diverge").

    ### Example 1 Starts ###
    * Question: What is the trend of the red curve between t=10 and t=25?
    * Ground Truth: increasing then decreasing
    * Response: The red curve is increasing between t=10 and t=25.

    {
        "extracted_answer": "increasing",
        "score": 0
    }
    ### Example 1 Ends ###

    ### Example 2 Starts ###
    * Question: What is the interval where the blue curve achieves the maximum value among [0, 50], [50, 100], [100, 150], and [150, 200]?
    * Ground Truth: [50, 100]
    * Response: The interval where the blue curve achieves the maximum value is [50, 100].

    {
        "extracted_answer": "[50, 100]",
        "score": 1
    }
    ### Example 2 Ends ###

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>

    """,
    3: """
    ### Rules ###
    * Give a score of 1 if and only if the two numbers are exactly equal in values. It's acceptable to have different notations (e.g., 0.01 and 10^-2; 1500 and 1.5e3).
    * Give a score of 0 if the two numbers are different in values.

    ### Example 1 Starts ###
    * Question: What is the value of the red curve at t=10?
    * Ground Truth: 0.01
    * Response: The value of the red curve at t=10 is 0.012.

    {
        "extracted_answer": "0.012",
        "score": 0
    }
    ### Example 1 Ends ###

    ### Example 2 Starts ###
    * Question: What is the value of the blue curve at t=50?
    * Ground Truth: 1500
    * Response: The value of the blue curve at t=50 is 1.5e3.

    {
        "extracted_answer": "1.5e3",
        "score": 1
    }
    ### Example 2 Ends ###

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>

    """,
    4: """
    ### Rules ###
    * Give a score of 1 if and only if the two numbers are exactly equal in values. It's acceptable to have different notations (e.g., 0.01 and 10^-2; 1500 and 1.5e3).
    * Give a score of 0 if the two numbers are different in values.

    ### Example 1 Starts ###
    * Question: What is the value of the red curve at t=10?
    * Ground Truth: 0.01
    * Response: The value of the red curve at t=10 is 0.012.

    {
        "extracted_answer": "0.012",
        "score": 0
    }
    ### Example 1 Ends ###

    ### Example 2 Starts ###
    * Question: What is the value of the blue curve at t=50?
    * Ground Truth: 1500
    * Response: The value of the blue curve at t=50 is 1.5e3.

    {
        "extracted_answer": "1.5e3",
        "score": 1
    }
    ### Example 2 Ends ###

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>

    """,
}

DOMAIN2ABBR = {
    "cs": "Computer Science",
    "econ": "Economics",
    "eess": "Electrical Engineering and Systems Science",
    "math": "Mathematics",
    "physics": "Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
}

NUM2YEAR = {"20": "2020", "21": "2021", "22": "2022", "23": "2023"}


def QNUM2QTYPE(qnum):
    if qnum in [1, 2, 3, 4, 5, 6, 7]:
        return "Information Extraction"
    elif qnum in [8, 9, 13, 14, 15]:
        return "Enumeration"
    elif qnum in [11, 16, 18]:
        return "Pattern Recognition"
    elif qnum in [10, 12, 19]:
        return "Counting"
    elif qnum in [17]:
        return "Compositionality"
    else:
        raise ValueError(f"Invalid qnum: {qnum}")


def NUMSUBPLOTS2SUBPLOTTYPE(num_subplots):
    if num_subplots == 1:
        return "1 Subplot"
    elif 2 <= num_subplots <= 4:
        return "2-4 Subplots"
    elif num_subplots >= 5:
        return "5+ Subplots"
    else:
        raise ValueError(f"Invalid num_subplots: {num_subplots}")


IDX2ANSTYPE = {
    1: "Text-in-Chart",
    2: "Text-in-General",
    3: "Number-in-Chart",
    4: "Number-in-General",
}

IDX2SRC = {1: "GPT-Sourced", 2: "GPT-Inspired", 3: "Completely Human"}


def D_TEMPLATE():
    return {
        "Overall Score": [],
        "By Question": {
            "Q1": [],
            "Q2": [],
            "Q3": [],
            "Q4": [],
            "Q5": [],
            "Q6": [],
            "Q7": [],
            "Q8": [],
            "Q9": [],
            "Q10": [],
            "Q11": [],
            "Q12": [],
            "Q13": [],
            "Q14": [],
            "Q15": [],
            "Q16": [],
            "Q17": [],
            "Q18": [],
            "Q19": [],
        },
        "By Category": {
            "Information Extraction": [],
            "Enumeration": [],
            "Pattern Recognition": [],
            "Counting": [],
            "Compositionality": [],
        },
        "By Subplot": {
            "1 Subplot": [],
            "2-4 Subplots": [],
            "5+ Subplots": [],
        },
        "By Subject": {
            "Computer Science": [],
            "Economics": [],
            "Electrical Engineering and Systems Science": [],
            "Mathematics": [],
            "Physics": [],
            "Quantitative Biology": [],
            "Quantitative Finance": [],
            "Statistics": [],
        },
        "By Year": {
            "2020": [],
            "2021": [],
            "2022": [],
            "2023": [],
        },
        "N_valid": [],
        "N_invalid": [],
    }


def R_TEMPLATE():
    return {
        "Overall Score": [],
        "By Answer Type": {
            "Text-in-Chart": [],
            "Text-in-General": [],
            "Number-in-Chart": [],
            "Number-in-General": [],
        },
        "By Source": {
            "GPT-Sourced": [],
            "GPT-Inspired": [],
            "Completely Human": [],
        },
        "By Subject": {
            "Computer Science": [],
            "Economics": [],
            "Electrical Engineering and Systems Science": [],
            "Mathematics": [],
            "Physics": [],
            "Quantitative Biology": [],
            "Quantitative Finance": [],
            "Statistics": [],
        },
        "By Year": {
            "2020": [],
            "2021": [],
            "2022": [],
            "2023": [],
        },
        "By Subplot": {
            "1 Subplot": [],
            "2-4 Subplots": [],
            "5+ Subplots": [],
        },
        "N_valid": [],
        "N_invalid": [],
    }


def get_rubric(qid):
    instruction = None
    if qid in [1]:
        instruction = DESCRIPTIVE_GRADING_ICL["title"]
    if qid in [2, 3, 4, 5, 6, 7]:
        instruction = DESCRIPTIVE_GRADING_ICL["ocr"]
    if qid in [8, 9, 10, 12, 14, 15, 17, 19]:
        instruction = DESCRIPTIVE_GRADING_ICL["quant"]
    if qid in [11]:
        instruction = DESCRIPTIVE_GRADING_ICL["bool"]
    if qid in [13]:
        instruction = DESCRIPTIVE_GRADING_ICL["enum"]
    if qid in [16]:
        instruction = DESCRIPTIVE_GRADING_ICL["trend"]
    if qid in [18]:
        instruction = DESCRIPTIVE_GRADING_ICL["layout"]
    assert instruction is not None, f"Instruction for qid {qid} is not found."
    return instruction


def build_json_keys(length):
    keys = []
    # specify the keys for gpt-4o's json response
    for i in range(1, length + 1):
        keys.append(f"extract_answer_T{i}")
        keys.append(f"score_T{i}")
    return str(keys)


def populate_grading_inputs(batch):
    query = ""
    for i, (_, response, answer) in enumerate(batch):
        # index, response, answer
        curr_query = "T{}:\nResponse {}: {}\nGround Truth {}: {}\n\n".format(
            i + 1, i + 1, response, i + 1, answer
        )
        query += curr_query
    return query


def verify_grading_output(data, length_data):
    # check the integrity of keys and values
    for i in range(1, length_data + 1):
        assert (
            f"extract_answer_T{i}" in data
        ), f"extract_answer_T{i} is not found in {data}"
        assert f"score_T{i}" in data, f"score_T{i} is not found in {data}"
        assert data[f"score_T{i}"] in [0, 1], f"score_T{i} is not in [0, 1]"
    return True


def build_dummy_output(length_data):
    # if failed to parse the response, return dummy data
    data = {}
    for i in range(1, length_data + 1):
        data[f"extract_answer_T{i}"] = "Failed to parse response"
        data[f"score_T{i}"] = -1
    return data


def preprocess_descriptive_grading_queries(annotations, predictions, num_templates=19):
    # group the responses based on the template id instead of figure id
    groups = {i: [] for i in range(1, num_templates + 1)}
    for pred in predictions:
        question_id = pred["question_id"]
        gt = annotations[question_id]
        if gt["question_type"] != "descriptive":
            continue
        resp_key = gt["resp_key"]
        qid = gt["qid"]
        answer = gt["answer"]
        groups[qid].append((resp_key, pred["answer"], answer))
    return groups


def build_descriptive_grading_queries(groups: Dict, nq_per_query: int = 5) -> List:
    queries = []
    for qid, data in groups.items():
        # batched evaluation based on number of questions per query (nq_per_query)
        for i in range(0, len(data), nq_per_query):
            # batch: list of tuples (resp_key, response, answer)
            batch = data[i : i + nq_per_query]
            # question based on the template id
            question = DESCRIPTIVE_GRADING_QMAP[qid]
            # build the json keys for GPT-4o's response
            json_keys = build_json_keys(len(batch))
            # populate batch size, question, and json keys spec
            prefix = (
                DESCRIPTIVE_GRADING_PREFIX.replace("<|NUM_TRIPLETS|>", str(len(batch)))
                .replace("<|OVERARCHING_QUESTION|>", question)
                .replace("<|JSON_KEYS|>", json_keys)
            )
            # add in-context grading example based on the template id
            rubric_icl = get_rubric(qid)
            # prompt + example + model responses
            grading_query = prefix + rubric_icl + populate_grading_inputs(batch)
            curr_query = {
                "resp_keys": [d[0] for d in batch],
                "grading_query": grading_query,
            }
            queries.append(curr_query)
    return queries


def postprocess_descriptive_grading_queries(queries: Dict) -> Dict:
    scores = {}
    for query in queries:
        # query contains resp_keys, grading_query, extract_answer and score
        resp_keys = query["resp_keys"]
        for i, resp_key in enumerate(resp_keys):
            # extract the answer and score for each response key
            extracted_answer = query[f"extract_answer_T{i+1}"]
            score = query[f"score_T{i+1}"]
            # store the extracted answer and score
            scores[resp_key] = {
                "resp_key": resp_key,
                "extracted_answer": extracted_answer,
                "score": score,
            }
    return scores


def build_reasoning_grading_queries(predictions: List[Dict], annotations: Dict) -> Dict:
    queries = {}
    for pred in predictions:
        question_id = pred["question_id"]
        gt = annotations[question_id]
        if gt["question_type"] != "reasoning":
            continue
        query = gt["raw_question"]
        response = pred["answer"]
        figure_id = str(gt["figure_id"])
        grading_query = REASONING_GRADING_PREFIX + deepcopy(
            REASONING_GRADING_INST[gt["inst_category"]]
        ).replace("<|question|>", query).replace(
            "<|ground_truth|>", gt["answer"]
        ).replace(
            "<|response|>", response
        )
        query = {
            "figure_id": figure_id,
            "grading_query": grading_query,
        }
        queries[figure_id] = query
    return queries


def get_descriptive_scores(predictions: List[Dict], annotations: Dict) -> Dict:
    stats = D_TEMPLATE()
    for pred in predictions:
        question_id = pred["question_id"]
        gt = annotations[question_id]
        if gt["question_type"] != "descriptive":
            continue
        qid = gt["qid"]
        score = pred["score"]
        if score not in [0, 1]:
            stats["N_invalid"].append(1)
            score = 0
        stats["N_valid"].append(1)
        stats["Overall Score"].append(score)
        stats["By Category"][QNUM2QTYPE(qid)].append(score)
        stats["By Subject"][DOMAIN2ABBR[gt["image_meta"]["category"]]].append(score)
        stats["By Year"][NUM2YEAR[gt["image_meta"]["year"]]].append(score)
        stats["By Subplot"][NUMSUBPLOTS2SUBPLOTTYPE(gt["num_subplots"])].append(score)
        stats["By Question"][f"Q{qid}"].append(score)

    stats["Question Type"] = "Descriptive"
    return stats


def get_reasoning_scores(predictions: List[Dict], annotations: Dict) -> Dict:
    stats = R_TEMPLATE()
    for pred in predictions:
        question_id = pred["question_id"]
        gt = annotations[question_id]
        if gt["question_type"] != "reasoning":
            continue
        num_subplot = gt["num_subplots"]
        subject = gt["image_meta"]["category"]
        year = gt["image_meta"]["year"]
        answer_type = gt["inst_category"]
        source = gt["qa_source"]
        score = pred["score"]

        if score not in [0, 1]:
            stats["N_invalid"].append(1)
            score = 0

        stats["N_valid"].append(1)
        stats["Overall Score"].append(score)
        stats["By Answer Type"][IDX2ANSTYPE[answer_type]].append(score)
        stats["By Source"][IDX2SRC[source]].append(score)
        stats["By Subject"][DOMAIN2ABBR[subject]].append(score)
        stats["By Year"][NUM2YEAR[year]].append(score)
        stats["By Subplot"][NUMSUBPLOTS2SUBPLOTTYPE(num_subplot)].append(score)
    stats["Question Type"] = "Reasoning"
    return stats


def get_stats(stats):
    if len(stats["N_valid"]) == 0:
        print("No valid scores")
        return
    for k, v in stats.items():
        # for sub categories
        if isinstance(v, dict):
            for k1, v1 in v.items():
                if len(v1) == 0:
                    print(f"{k1}: No valid scores")
                    stats[k][k1] = 0
                else:
                    stats[k][k1] = round(100 * sum(v1) / len(v1), 2)
        # metadata
        elif k == "Question Type":
            pass
        # for overall scores
        elif k not in ["N_valid", "N_invalid"]:
            if len(v) == 0:
                print(f"{k}: No valid scores")
                stats[k] = 0
            else:
                stats[k] = round(100 * sum(v) / len(v), 2)
        # for number of valid/invalid scores
        else:
            stats[k] = len(v)
    return stats


def get_descriptive_result_gpt(
    prompt: str, length: int, model: GPT, max_retries: int = 10
) -> Dict:
    curr_retries = 0
    max_tokens = 256
    temperature = 0
    while curr_retries < max_retries:
        try:
            message = model.build_message(query=prompt)
            response = model.infer(
                chat_messages=message,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                seed=42,
            )
            content = json.loads(response.content)
            verify_grading_output(content, length)
            break
        except Exception as e:
            print(f"Error: {e}")
            # increase the max_tokens if the response is too long
            if max_tokens >= 1024:
                print(f"Failed to get response for prompt: {prompt}")
                content = build_dummy_output(length)
                break
            else:
                max_tokens = min(1024, max_tokens * 2)  # double the max_tokens
                print(f"Retrying with max_tokens: {max_tokens}")
            temperature += 0.1
            # otherwise, retry the request
            curr_retries += 1

    # if failed to get response, return dummy data
    if curr_retries == max_retries:
        print(f"Failed to get response for prompt: {prompt}")
        content = build_dummy_output(length)
    return content


def get_reasoning_result_gpt(
    prompt: str, model: GPT, max_retries: int = 10
) -> Tuple[str, int]:
    curr_retries = 0
    max_tokens = 256
    while curr_retries < max_retries:
        try:
            message = model.build_message(query=prompt)
            response = model.infer(
                chat_messages=message,
                max_tokens=max_tokens,
                temperature=0,
                top_p=1,
                seed=42,
            )
            content = json.loads(response.content)
            ext, scr = content["extracted_answer"], content["score"]
            break
        except Exception as e:
            print(f"Error: {e}")
            # increase the max_tokens if the response is too long
            if max_tokens >= 1024:
                print(f"Failed to get response for prompt: {prompt}")
                ext, scr = "Failed to parse response", -1
                break
            else:
                max_tokens = min(1024, max_tokens * 2)  # double the max_tokens
                print(f"Retrying with max_tokens: {max_tokens}")
            # otherwise, retry the request
            curr_retries += 1
    # if failed to get response, return dummy data
    if curr_retries == max_retries:
        print(f"Failed to get response for prompt: {prompt}")
        ext, scr = "Failed to parse response", -1
    return ext, scr


def update_predictions(
    predictions: List[Dict], annotations: Dict, scores: Dict, type: str
) -> List[Dict]:
    for pred in predictions:
        question_id = pred["question_id"]
        gt = annotations[question_id]
        if gt["question_type"] != type:
            continue
        resp_key = str(gt.get("resp_key", gt["figure_id"]))
        pred.update(scores[resp_key])
    return predictions


def get_result(annotations: Dict, predictions: List[Dict], llm_evaluator: GPT) -> Dict:
    groups = preprocess_descriptive_grading_queries(annotations, predictions)

    # batched evaluation based on number of questions per query (nq_per_query)
    queries = build_descriptive_grading_queries(groups)
    combined_queries = []
    results = {}
    for query in queries:
        result = get_descriptive_result_gpt(
            query["grading_query"], len(query["resp_keys"]), llm_evaluator
        )

        # query contains resp_keys, grading_query, extract_answer and score
        combined_queries.append({**query, **result})

    queries = combined_queries
    # flatten the queries and only keep the necessary fields
    queries = postprocess_descriptive_grading_queries(queries)
    predictions = update_predictions(
        predictions, annotations, queries, type="descriptive"
    )
    # with open(osp.join(output_dir, "descriptive_grading.json"), "w") as f:
    #     json.dump(queries, f, indent=2)

    descriptive_stats = get_descriptive_scores(predictions, annotations)
    descriptive_stats = get_stats(descriptive_stats)
    results["descriptive"] = descriptive_stats
    # Process reasoning
    queries = build_reasoning_grading_queries(predictions, annotations)
    for figure_id, query in queries.items():
        ext, scr = get_reasoning_result_gpt(query["grading_query"], llm_evaluator)
        queries[figure_id]["extracted_answer"] = ext
        queries[figure_id]["score"] = scr
        queries[figure_id].pop("grading_query")

    predictions = update_predictions(
        predictions, annotations, queries, type="reasoning"
    )
    reasoning_stats = get_reasoning_scores(predictions, annotations)
    reasoning_stats = get_stats(reasoning_stats)
    # with open(osp.join(output_dir, "reasoning_grading.json"), "w") as f:
    #     json.dump(queries, f, indent=2)
    results["reasoning"] = reasoning_stats
    final_scores = []
    if results["descriptive"]:
        final_scores.append(results["descriptive"]["Overall Score"])
    if results["reasoning"]:
        final_scores.append(results["reasoning"]["Overall Score"])

    results["accuracy"] = round(sum(final_scores) / len(final_scores), 2)
    return results
