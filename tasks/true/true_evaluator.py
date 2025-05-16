from typing import Dict, List, Union
import unicodedata
from collections import defaultdict
from flagevalmm.evaluator import BaseEvaluator
from flagevalmm.registry import EVALUATORS


class SubstringEditDistance:
    def __init__(self, initial_size: int = 1024):
        """
        Initialize with a default matrix size

        Args:
            initial_size: Initial size for both dimensions of the dp matrix
        """
        self.size = initial_size
        self.dp = [[0] * initial_size for _ in range(initial_size)]

    def _resize_matrix(self, required_rows: int, required_cols: int):
        """
        Resize the dp matrix if current size is insufficient

        Args:
            required_rows: Number of rows needed
            required_cols: Number of columns needed
        """
        new_size = max(required_rows, required_cols)
        if new_size > self.size:
            # Double the size until it's sufficient
            while self.size < new_size:
                self.size *= 2
            # Create new matrix
            self.dp = [[0] * self.size for _ in range(self.size)]

    def calculate(self, source: str, target: str) -> tuple[int, str]:
        """
        Calculate minimal edit distance between target string and any substring of source string

        Args:
            source: The main text to search within (e.g., OCR output or document text)
            target: The text pattern to search for (e.g., ground truth or query text)

        Returns:
            A tuple of (minimal edit distance, best matching substring from source)
        """
        m, n = len(source), len(target)

        # Resize matrix if necessary
        self._resize_matrix(n + 1, m + 1)

        # Initialize first row
        for j in range(m + 1):
            self.dp[0][j] = 0

        # Initialize first column
        for i in range(1, n + 1):
            self.dp[i][0] = i

        # Fill the dp table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if target[i - 1] == source[j - 1]:
                    self.dp[i][j] = self.dp[i - 1][j - 1]
                else:
                    self.dp[i][j] = min(
                        self.dp[i - 1][j - 1] + 1,  # replace
                        self.dp[i - 1][j] + 1,  # delete
                        self.dp[i][j - 1] + 1,  # insert
                    )

        # Find minimum in the last row and its position
        min_dist = float("inf")
        end_pos = 0
        for j in range(m + 1):
            if self.dp[n][j] < min_dist:
                min_dist = self.dp[n][j]
                end_pos = j

        # Backtrack to find the start position
        start_pos = end_pos
        curr_row = n
        curr_col = end_pos

        while curr_row > 0:
            candidates = [
                (curr_row - 1, curr_col - 1),  # diagonal
                (curr_row - 1, curr_col),  # up
                (curr_row, curr_col - 1),  # left
            ]

            next_pos = min(
                (pos for pos in candidates if pos[1] >= 0),
                key=lambda pos: self.dp[pos[0]][pos[1]],
            )

            if next_pos[1] < curr_col:
                start_pos = next_pos[1]

            curr_row, curr_col = next_pos

        return min_dist, source[start_pos:end_pos]

    def get_matrix_size(self) -> int:
        """
        Returns current matrix size
        """
        return self.size


@EVALUATORS.register_module()
class TRUEEvaluator(BaseEvaluator):
    def __init__(
        self,
        anls_threshold: float = 0.95,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.anls_threshold = anls_threshold
        self.sub_edit_distance = SubstringEditDistance()
        self.use_anls_types = ["Full Page", "Chalkboard", "Digital Notes"]

    def get_score(self, gt: Dict, pred: Dict) -> Union[float, List[float]]:
        dataset_name = gt.get("dataset", "")
        gt_ans = gt["answer"]
        question_type = gt["question_type"]
        question_subtype = gt.get("question_subtype", "")
        # Get answer type
        answer_type = "single_answer"
        if isinstance(gt_ans, dict):
            if "answer_type" in gt_ans:
                answer_type = gt_ans["answer_type"]
            if "answer_items" in gt_ans:
                gt_ans = gt_ans["answer_items"]

        # Convert single answer to list for unified processing
        answers = gt_ans if isinstance(gt_ans, list) else [gt_ans]

        def process(x, enable_normalize=True):
            if enable_normalize:
                x = unicodedata.normalize("NFKC", x)
            x = x.strip().replace("\n", " ")
            if (
                question_type == "Digit String Recognition"
                or dataset_name == "HME100k"
                or question_subtype == "Digit String Recognition"
            ):
                return x.replace(" ", "")
            else:
                return x.lower()

        # Check if any answer variant matches the prediction as a substring
        def match_any(pred_str, answer_variants):
            return int(any(process(answer) in pred_str for answer in answer_variants))

        pred_ans = pred["answer"]

        if question_subtype in self.use_anls_types:
            processed_pred = process(pred_ans)
            matched_strs = []
            # Modified Average Average Normalized Levenshtein Similarity(ANLS) for blocks
            distance_sum, matched_len_sum, answers_len_sum = 0, 0, 0

            # For unordered items, concatenate all items
            if answer_type == "unordered_items":
                concated_answer = ""
                for answer in answers:
                    concated_answer += answer[0] + " "
                answers = [concated_answer.strip()]

            for answer in answers:
                answers_len_sum += len(answer)
                distance, matched_str = self.sub_edit_distance.calculate(
                    processed_pred, process(answer)
                )
                distance_sum += distance
                matched_len_sum += len(matched_str)
                matched_strs.append([matched_str, distance])
            pred["matched_str"] = matched_strs
            anls = 1 - distance_sum / max(matched_len_sum, answers_len_sum)
            pred["anls"] = anls
            return anls > self.anls_threshold
        else:
            processed_pred = process(pred_ans)
            if answer_type == "unordered_items":
                return int(
                    all(
                        match_any(processed_pred, item_variants)
                        for item_variants in answers
                    )
                )
            # Otherwise, keep matching a single list of answer variants
            return match_any(processed_pred, answers)

    def cal_accuracy(
        self, annotations: Dict, predictions: List[Dict], *args, **kwargs
    ) -> Dict:
        class ScoreTracker:
            def __init__(self):
                self.total_score = 0
                self.count = 0
                self.accuracy = 0
                self.subtypes = defaultdict(
                    lambda: [0, 0, 0]
                )  # [score_sum, count, accuracy]

        results = {}
        scores_by_type = defaultdict(ScoreTracker)
        for pred in predictions:
            question_id = str(pred["question_id"])
            gt = annotations[question_id]
            score = self.get_score(gt, pred)

            if isinstance(score, list):
                pred["score_list"] = score
                score = sum(score) / len(score)
            pred.update(
                {
                    "score": score,
                    "label": gt["answer"],
                    "question_type": gt["question_type"],
                    "question_subtype": gt.get("question_subtype", "Default"),
                }
            )

            # Update scores
            tracker = scores_by_type[pred["question_type"]]
            tracker.total_score += score
            tracker.count += 1
            tracker.subtypes[pred["question_subtype"]][0] += score
            tracker.subtypes[pred["question_subtype"]][1] += 1
        # Calculate accuracy
        for tracker in scores_by_type.values():
            tracker.accuracy = round(tracker.total_score / tracker.count, 3)
            for sub_type in tracker.subtypes:
                tracker.subtypes[sub_type][2] = round(
                    tracker.subtypes[sub_type][0] / tracker.subtypes[sub_type][1], 3
                )
        final_score = sum(tracker.total_score for tracker in scores_by_type.values())
        results["final_score"] = [final_score, len(predictions)]
        results["accuracy"] = round(final_score / len(predictions) * 100, 3)

        # Convert ScoreTracker objects to the expected format
        for qtype, tracker in scores_by_type.items():
            results[qtype] = [
                tracker.total_score,
                tracker.count,
                tracker.accuracy,
                dict(tracker.subtypes),
            ]

        return results
