import re
import json
import os.path as osp
from typing import Dict, List
from flagevalmm.registry import EVALUATORS
from flagevalmm.evaluator import BaseEvaluator


@EVALUATORS.register_module()
class CocoEvaluator(BaseEvaluator):
    """
    The evaluation method is adapted from the reliable_vqa project
    (https://github.com/facebookresearch/reliable_vqa/blob/main/eval_scripts/reliable_vqa_eval.py)
    with modifications to improve robustness and adapt to the flagevalmm framework.
    """

    def __init__(self, keep_digital=2, **kwargs):
        super().__init__(**kwargs)
        self.keep_digital = keep_digital
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.number_map = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.period_strip = re.compile("(?!<=\\d)(\\.)(?!\\d)")
        self.comma_strip = re.compile("(\\d)(\\,)(\\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def preporcess_data(self, data):
        data = data.replace("\n", " ")
        data = data.replace("\t", " ")
        data = data.strip().lower()
        data = self.process_punctuation(data)
        data = self.process_digit_article(data)
        return data

    def cal_accuracy(
        self, annotations: Dict, predictions: List[Dict], *args, **kwargs
    ) -> Dict:
        acc_qa = []
        for prediction in predictions:
            question_id = str(prediction["question_id"])
            gts = annotations[question_id]
            gts["answer"] = [self.preporcess_data(gt) for gt in gts["answer"]]
            res_ans = self.preporcess_data(prediction["answer"])
            gt_acc = []

            for i in range(len(gts["answer"])):
                other_gt_ans = [item for j, item in enumerate(gts["answer"]) if j != i]
                matching_ans = [item for item in other_gt_ans if item == res_ans]
                acc = min(1, float(len(matching_ans)) / 3)
                gt_acc.append(acc)
            avg_gt_acc = float(sum(gt_acc)) / len(gt_acc)
            acc_qa.append(avg_gt_acc)
            prediction["correct"] = avg_gt_acc
            prediction["label"] = gts["answer"]

        results = {
            "accuracy": round(100 * float(sum(acc_qa)) / len(acc_qa), self.keep_digital)
        }
        return results

    def process(self, dataset, output_dir, **kwargs):
        """
        Args:
            dataset (Dataset): dataset instance
            answers (list): list of answers
        """
        annotation = dataset.get_annotation()
        dataset_name = dataset.name
        result_file = osp.join(output_dir, dataset_name + ".json")
        predictions = json.load(open(result_file))
        results = {}
        predictions, filtered_predictions = self.filter_rejected(predictions, results)
        results.update(self.cal_accuracy(annotation, predictions))
        self.save(results, predictions + filtered_predictions, dataset_name, output_dir)
        return results

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.punct:
            condition1 = p + " " in in_text or " " + p in in_text
            condition2 = re.search(self.comma_strip, in_text) is not None
            if condition1 or condition2:
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.period_strip.sub("", out_text, re.UNICODE)
        return out_text

    def preprocess_string(self, in_text):
        if len(in_text.split()) == 0:
            return in_text
        first_word = in_text.split()[0]
        first_word = first_word.replace(",", "")
        first_word = first_word.replace(".", "")
        if first_word in [
            "no",
            "none",
            "yes",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        ]:
            return first_word
        sentence_to_convert = {"not": "no", "unable to": "unknown"}
        for sent in sentence_to_convert:
            if sent in in_text:
                return sentence_to_convert[sent]
        return in_text

    def process_digit_article(self, in_text):
        out_text = []
        in_text = self.preprocess_string(in_text.lower())
        temp_text = in_text.split()
        for word in temp_text:
            word = self.number_map.setdefault(word, word)
            if word not in self.articles:
                out_text.append(word)
            else:
                pass
        for i, word in enumerate(out_text):
            if word in self.contractions:
                out_text[i] = self.contractions[word]
        out_text = " ".join(out_text)
        return out_text
