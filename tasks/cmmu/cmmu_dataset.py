import json
import glob
import os.path as osp

from flagevalmm.dataset.vqa_base_dataset import VqaBaseDataset
from flagevalmm.registry import DATASETS, PROMPTS
from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR


@DATASETS.register_module()
class CmmuDataset(VqaBaseDataset):
    def __init__(
        self,
        name,
        data_root=None,
        cache_dir=FLAGEVALMM_DATASETS_CACHE_DIR,
        config=None,
        prompt_template=None,
        debug=False,
        with_label=False,
        shift_check=True,
        filter_types=None,
        **kwargs,
    ):
        if data_root is not None:
            self.data_root = data_root
        else:
            # get data from cache
            cache_dir = osp.expanduser(cache_dir)
            self.data_root = osp.join(cache_dir, config.processed_dataset_path)
        self.name = name
        if prompt_template is not None:
            prompt_template = PROMPTS.build(prompt_template)
        self.prompt_template = prompt_template
        self.with_label = with_label or debug
        self.shift_check = shift_check
        if isinstance(filter_types, list):
            self.filter_types = set(filter_types)
        else:
            self.filter_types = None
        self.annotations = self.load_data()
        if debug:
            self.annotations = self.annotations[:32]

    def __len__(self):
        return len(self.annotations)

    def build_prompt(self, annotation):
        question = annotation["question_info"]
        if "options" in annotation:
            choices = annotation["options"]
            base = ord("A")
            for i, choice in enumerate(choices):
                question += "\n" + chr(base + i) + ". " + choice
        if "sub_questions" in annotation:
            sub_questions = annotation["sub_questions"]
            question += " " + sub_questions[0]
        question = "<image 1>" + question
        if self.prompt_template is not None:
            question = self.prompt_template.build_prompt(
                question=question,
                question_type=annotation["type"],
                subject=annotation["subject"],
            )
        return question

    def __getitem__(self, index):
        annotation = self.annotations[index]
        question = self.build_prompt(annotation)
        ret = {
            "img_path": [osp.join(self.data_root, annotation["images"][0])],
            "question": question,
            "question_id": annotation["id"],
            "type": annotation["type"],
        }
        if self.with_label:
            ret["label"] = annotation["answer"]
        return ret

    def meta_info(self):
        return {"name": self.name, "length": len(self.annotations), "type": "vqa"}

    def process_fill_in_blank(self, data):
        splited_data = []
        for i, sub_question in enumerate(data["sub_questions"]):
            sub_data = data.copy()
            sub_data["id"] = f"{data['id']}-{i}"
            sub_data["question_info"] += " " + sub_question
            sub_data["answer"] = sub_data["answer"][i]
            del sub_data["sub_questions"]
            splited_data.append(sub_data)
        return splited_data

    def process_multiple_choice(self, data):
        if not self.shift_check:
            return [data]
        # circular shift choices and update answer
        shifted_data = []
        choices = data["options"]
        base = ord("A")
        for i in range(len(choices)):
            new_data = data.copy()
            new_data["id"] = f"{data['id']}-{i}"
            # shift right ABCD -> DABC
            new_data["options"] = choices[-i:] + choices[:-i]
            new_data["answer"] = chr(
                ord("A") + (ord(new_data["answer"]) - base + i) % len(choices)
            )
            shifted_data.append(new_data)

        return shifted_data

    def load_data(self):
        annotations = []
        for anno_file in glob.glob(osp.join(self.data_root, "**", "*.jsonl")):
            with open(anno_file) as fin:
                for line in fin:
                    data = json.loads(line)
                    if (
                        self.filter_types is not None
                        and data["type"] not in self.filter_types
                    ):
                        continue
                    if data["type"] == "fill-in-the-blank":
                        annotations += self.process_fill_in_blank(data)
                    elif data["type"] == "multiple-choice":
                        annotations += self.process_multiple_choice(data)
                    else:
                        annotations.append(data)

        return annotations

    def get_annotation(self):
        anno_dict = {}  # question_id: answer_dict
        for anno in self.annotations:
            if anno["id"] in anno_dict:
                print(anno["id"])
            anno_dict[anno["id"]] = anno
        return anno_dict
