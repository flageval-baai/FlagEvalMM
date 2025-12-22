import os
import os.path as osp
import re
import time
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

from flagevalmm.common.logger import get_logger
from flagevalmm.models.http_client import HttpClient
from flagevalmm.registry import EVALUATORS


logger = get_logger(__name__)

GROUPS = [
    "background_change",
    "color_alter",
    "material_alter",
    "motion_change",
    "ps_human",
    "style_change",
    "subject-add",
    "subject-remove",
    "subject-replace",
    "text_change",
    "tone_transfer",
]


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(sum(values) / len(values)), 4)


def _resize_to_area(img: Image.Image, target_area: int) -> Image.Image:
    ratio = img.width / img.height
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return img.resize((int(width), int(height)))


# === TIE prompts extracted from viescore ===
_TIE_CONTEXT = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{
"score" : [...],
"reasoning" : "..."
}"""

_TIE_TWO_IMAGE_RULE = """RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit."""

_TIE_SC_RULE = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

First lets look at the first set of input (1st and 2nd images) as an example. 
Editing instruction: What if the man had a hat?
Output:
{
"score" : [5, 10],
"reasoning" :  "The hat exists but does not suit well. The hat also looks distorted. But it is a good edit because only a hat is added and the background is persevered."
}

Now evaluate the second set of input (3th, 4th images).
Editing instruction: <instruction>"""

_TIE_PQ_RULE = """RULES:

One image will be provided; The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on image naturalness. 
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts. 
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]


First lets look at the first set of input (1st image) as an example. 
Output:
{
"score" : [5, 5],
"reasoning" :  "The image gives an unnatural feeling on hands of the girl. There is also minor distortion on the eyes of the girl."
}

Now evaluate the second set of input (2nd image).

"""


def _load_key_from_file(key_path: Optional[str]) -> Optional[str]:
    if not key_path or not osp.exists(key_path):
        return None
    try:
        with open(key_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() in {"OPENAI_API_KEY", "BAAI_OPENAI_API_KEY"}:
                        return v.strip()
    except Exception as err:  # pragma: no cover - best effort helper
        logger.debug("Failed to load api_key from %s: %s", key_path, err)
    return None


def _extract_scores(text: str, expected_len: int) -> Optional[List[float]]:
    """Parse model output into a list of scores in [0,10]."""
    if not text:
        return None
    # Try to find a JSON/list block first.
    match = re.search(r"\[([^\]]+)\]", text)
    numbers: List[float] = []
    if match:
        candidates = re.split(r"[,\s]+", match.group(1).strip())
        for item in candidates:
            if not item:
                continue
            try:
                val = float(item)
                if 0 <= val <= 10:
                    numbers.append(val)
            except ValueError:
                continue
    # Fallback: look for scoreX: N patterns.
    if not numbers:
        for val in re.findall(r"score\d*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I):
            try:
                num = float(val)
                if 0 <= num <= 10:
                    numbers.append(num)
            except ValueError:
                continue
    # Last resort: any number between 0 and 10.
    if not numbers:
        for val in re.findall(r"\b([0-9]+(?:\.[0-9]+)?)\b", text):
            try:
                num = float(val)
                if 0 <= num <= 10:
                    numbers.append(num)
            except ValueError:
                continue
    if not numbers or len(numbers) < expected_len:
        return None
    return numbers[:expected_len]


def _extract_think(text: str) -> Optional[str]:
    """Extract <think>...</think> content from model output if present."""
    if not text:
        return None
    match = re.search(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE)
    if not match:
        return None
    content = match.group(1).strip()
    return content or None


class _TieVIEScorer:
    """Self-contained tie scorer using the extracted VIE prompts."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str],
        url: Optional[str],
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_image_size: Optional[int] = None,
        min_short_side: Optional[int] = None,
        max_long_side: Optional[int] = None,
        retry_time: Optional[int] = None,
        use_cache: bool = True,
    ) -> None:
        self.client = HttpClient(
            model_name=model_name,
            api_key=api_key,
            url=url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_image_size=max_image_size,
            min_short_side=min_short_side,
            max_long_side=max_long_side,
            retry_time=retry_time,
            use_cache=True,
        )

        self.sc_prompt_prefix = "\n".join(
            [_TIE_CONTEXT, _TIE_TWO_IMAGE_RULE, _TIE_SC_RULE]
        )
        self.pq_prompt = "\n".join([_TIE_CONTEXT, _TIE_PQ_RULE])

    def evaluate(
        self, images: List[Image.Image], instruction: str
    ) -> Tuple[float, float, float]:
        sem_score, qual_score, overall, _sc_resp, _pq_resp = self.evaluate_with_raw(
            images, instruction
        )
        return sem_score, qual_score, overall

    def evaluate_with_raw(
        self, images: List[Image.Image], instruction: str
    ) -> Tuple[float, float, float, str, str]:
        if len(images) < 2:
            raise ValueError("TIE scoring requires two images (source + edited).")
        sc_prompt = self.sc_prompt_prefix.replace("<instruction>", instruction)
        sc_messages = self.client.build_message(
            query=sc_prompt, multi_modal_data={"image": images}
        )
        sc_resp = self.client.infer(sc_messages).content
        sc_scores = _extract_scores(sc_resp, expected_len=2)
        if not sc_scores:
            raise ValueError(f"Failed to parse semantics scores: {sc_resp}")

        pq_messages = self.client.build_message(
            query=self.pq_prompt, multi_modal_data={"image": [images[-1]]}
        )
        pq_resp = self.client.infer(pq_messages).content
        pq_scores = _extract_scores(pq_resp, expected_len=2)
        if not pq_scores:
            raise ValueError(f"Failed to parse quality scores: {pq_resp}")

        sem_score = float(min(sc_scores))
        qual_score = float(min(pq_scores))
        overall = float(math.sqrt(sem_score * qual_score))
        return sem_score, qual_score, overall, str(sc_resp), str(pq_resp)


@EVALUATORS.register_module()
class GEditEvaluator:
    """GEdit-Bench evaluator using inlined VIE tie prompts (semantics / quality / overall).

    This mirrors the reference script at tasks/t2i/gedit/evaluate.py and
    aggregates scores by task_type plus Intersection_exist split.
    """

    def __init__(
        self,
        data_root: str,
        image_dir: Optional[str] = None,
        input_image_key: str = "input_image_raw",
        fallback_input_key: str = "input_image",
        instruction_key: str = "instruction",
        language_key: str = "instruction_language",
        task_type_key: str = "task_type",
        intersection_key: str = "Intersection_exist",
        language: str = "all",
        api_key: str = None,
        model: Optional[str] = None,
        url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_image_size: Optional[int] = None,
        min_short_side: Optional[int] = None,
        max_long_side: Optional[int] = None,
        retry_time: Optional[int] = None,
        max_workers: int = 6,
        resize_area: int = 512 * 512,
        max_retries: int = 3,
        base_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not osp.isabs(data_root) and base_dir:
            data_root = osp.join(base_dir, data_root)
        self.data_root = data_root
        self.image_root = (
            osp.join(self.data_root, image_dir) if image_dir else self.data_root
        )
        self.input_image_key = input_image_key
        self.fallback_input_key = fallback_input_key
        self.instruction_key = instruction_key
        self.language_key = language_key
        self.task_type_key = task_type_key
        self.intersection_key = intersection_key
        self.languages = (
            ["en", "cn"] if language.lower() == "all" else [language.lower()]
        )
        self.max_workers = max_workers
        self.resize_area = resize_area
        self.max_retries = max_retries

        self.model_name = model
        self.api_key = api_key
        self.url = url
        self.api_key = api_key
        self.tie_scorer = _TieVIEScorer(
            model_name=self.model_name,
            api_key=self.api_key,
            url=self.url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_image_size=max_image_size,
            min_short_side=min_short_side,
            max_long_side=max_long_side,
            retry_time=retry_time,
        )

    def _resolve_path(self, path: str) -> str:
        return path if osp.isabs(path) else osp.join(self.image_root, path)

    def _resolve_source_path(self, ann: Dict[str, Any]) -> Optional[str]:
        path = ann.get(self.input_image_key) or ann.get(self.fallback_input_key)
        if path is None:
            return None
        return self._resolve_path(path)

    def _resolve_edited_path(self, image_entry: Any, output_dir: str) -> Optional[str]:
        if image_entry is None:
            return None
        if isinstance(image_entry, list):
            if not image_entry:
                return None
            image_entry = image_entry[0]
        if not isinstance(image_entry, str):
            return None
        return image_entry if osp.isabs(image_entry) else osp.join(
            output_dir, "samples", image_entry
        )

    def _load_and_resize(self, path: str) -> Image.Image:
        if not osp.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        img = Image.open(path).convert("RGB")
        return _resize_to_area(img, self.resize_area)

    def _score_single(
        self,
        qid: str,
        prompt: str,
        lang: str,
        task_type: str,
        intersection: bool,
        source_path: str,
        edited_path: str,
    ) -> Optional[Dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                source_img = self._load_and_resize(source_path)
                edited_img = self._load_and_resize(edited_path)
                sem, qual, overall, sc_resp, pq_resp = self.tie_scorer.evaluate_with_raw(
                    [source_img, edited_img], prompt
                )
                return {
                    "id": qid,
                    "task_type": task_type,
                    "language": lang,
                    "intersection": bool(intersection),
                    "semantics_score": float(sem),
                    "quality_score": float(qual),
                    "overall_score": float(overall),
                    "sc_resp": sc_resp,
                    "sc_think": _extract_think(sc_resp),
                    "pq_think": _extract_think(pq_resp),
                    "edited_image": edited_path,
                    "source_image": source_path,
                }
            except Exception as err:  # pragma: no cover - runtime robustness
                wait = (attempt + 1) * 2
                logger.warning(
                    "GEditEvaluator retry %s/%s for %s (%s): %s",
                    attempt + 1,
                    self.max_retries,
                    qid,
                    task_type,
                    err,
                )
                if attempt + 1 >= self.max_retries:
                    return None
                time.sleep(wait)
        return None

    def _aggregate(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        semantics = defaultdict(list)
        quality = defaultdict(list)
        overall = defaultdict(list)
        semantics_inter = defaultdict(list)
        quality_inter = defaultdict(list)
        overall_inter = defaultdict(list)

        for rec in records:
            group = rec.get("task_type", "unknown")
            semantics[group].append(rec["semantics_score"])
            quality[group].append(rec["quality_score"])
            overall[group].append(rec["overall_score"])
            if rec.get("intersection", False):
                semantics_inter[group].append(rec["semantics_score"])
                quality_inter[group].append(rec["quality_score"])
                overall_inter[group].append(rec["overall_score"])

        def _avg_map(stat: Dict[str, List[float]]) -> Dict[str, Optional[float]]:
            return {k: _mean(v) for k, v in stat.items() if v}

        return (
            _avg_map(semantics),
            _avg_map(quality),
            _avg_map(overall),
            _avg_map(semantics_inter),
            _avg_map(quality_inter),
            _avg_map(overall_inter),
        )

    def get_metric_results(
        self, output_info: List[Dict[str, Any]], output_dir: str, annotations: Dict, **kwargs
    ) -> Dict[str, Any]:
        tasks = []
        for info in output_info:
            qid_raw = info.get("id") or info.get("question_id")
            if qid_raw is None:
                continue
            qid = str(qid_raw)
            ann = annotations.get(qid)
            if ann is None:
                logger.warning("Annotation missing for %s, skip.", qid)
                continue

            lang = str(ann.get(self.language_key, "")).lower()
            if self.languages != ["en", "cn"] and lang not in self.languages:
                continue

            prompt = str(
                ann.get(self.instruction_key)
                or ann.get("prompt")
                or info.get("prompt", "")
            )
            source_path = self._resolve_source_path(ann)
            edited_path = self._resolve_edited_path(info.get("images"), output_dir)
            if not source_path or not edited_path:
                logger.warning("Missing paths for %s, skip.", qid)
                continue

            tasks.append(
                (
                    qid,
                    prompt,
                    lang,
                    ann.get(self.task_type_key, "unknown"),
                    bool(ann.get(self.intersection_key, False)),
                    source_path,
                    edited_path,
                )
            )

        results: List[Dict[str, Any]] = []
        if not tasks:
            logger.warning("GEditEvaluator: no valid tasks to evaluate.")
            return {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self._score_single, *task): task[0] for task in tasks
            }
            for fut in tqdm(
                as_completed(future_to_id),
                total=len(future_to_id),
                desc="GEdit evaluating",
            ):
                res = fut.result()
                if res:
                    results.append(res)

        result_by_id = {r["id"]: r for r in results}
        for info in output_info:
            qid = str(info.get("id") or info.get("question_id") or "")
            rec = result_by_id.get(qid)
            if not rec:
                continue
            info["gedit_semantics_score"] = rec["semantics_score"]
            info["gedit_quality_score"] = rec["quality_score"]
            info["gedit_overall_score"] = rec["overall_score"]
            info["gedit_task_type"] = rec.get("task_type")
            info["gedit_language"] = rec.get("language")
            info["gedit_intersection"] = rec.get("intersection", False)
            info["gedit_sc_resp"] = rec.get("sc_resp")
            info["gedit_sc_think"] = rec.get("sc_think")
            info["gedit_pq_think"] = rec.get("pq_think")
            info["gedit_edited_image"] = rec.get("edited_image")
            info["gedit_source_image"] = rec.get("source_image")

        (
            sem_map,
            qual_map,
            overall_map,
            sem_map_inter,
            qual_map_inter,
            overall_map_inter,
        ) = self._aggregate(results)

        sem_all = [r["semantics_score"] for r in results]
        qual_all = [r["quality_score"] for r in results]
        overall_all = [r["overall_score"] for r in results]
        inter_all = [r for r in results if r.get("intersection")]

        res_dict: Dict[str, Any] = {
            "gedit_semantics_mean": _mean(sem_all),
            "gedit_quality_mean": _mean(qual_all),
            "gedit_overall_mean": _mean(overall_all),
            "gedit_semantics_by_group": sem_map,
            "gedit_quality_by_group": qual_map,
            "gedit_overall_by_group": overall_map,
            "gedit_semantics_by_group_intersection": sem_map_inter,
            "gedit_quality_by_group_intersection": qual_map_inter,
            "gedit_overall_by_group_intersection": overall_map_inter,
            "gedit_intersection_mean_semantics": _mean(
                [r["semantics_score"] for r in inter_all]
            ),
            "gedit_intersection_mean_quality": _mean(
                [r["quality_score"] for r in inter_all]
            ),
            "gedit_intersection_mean_overall": _mean(
                [r["overall_score"] for r in inter_all]
            ),
            "gedit_language_filter": self.languages,
            "gedit_sample_count": len(results),
            "gedit_intersection_count": len(inter_all),
        }
        return res_dict
