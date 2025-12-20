import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from flagevalmm.common.logger import get_logger
from flagevalmm.models.api_response import ApiResponse
from flagevalmm.models.http_client import HttpClient
from flagevalmm.prompt.prompt_tools import encode_image
from flagevalmm.registry import EVALUATORS

logger = get_logger(__name__)


def _clean_text(text: str) -> str:
    cleaned = text or ""
    keywords = ["addCriterion", "No text recognized."]
    for keyword in keywords:
        cleaned = cleaned.replace(keyword, "").replace(f"\n{keyword}", "").replace(
            f"{keyword}\n", ""
        )
    return cleaned.strip()


def _normalize(text: str) -> str:
    return "".join(text.lower().split())


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _preprocess_string(s: str, mode: str = "en") -> str:
    cleaned = re.sub(
        r"[^\u4e00-\u9fa5a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", "", s
    )
    if mode == "en":
        normalized = re.sub(r"\s+", " ", cleaned)
        return normalized.strip().lower()
    pattern = re.compile(
        r"[\u4e00-\u9fa5a-zA-Z0-9àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]"
    )
    return "".join(pattern.findall(s)).strip()


def _counter2list(counter: Counter) -> List[str]:
    return [item for item, count in counter.items() for _ in range(count)]


def _calculate_char_match_ratio(
    text_gt: str, ocr_str: str, mode: str = "en"
) -> Tuple[List[str], Any, List[str]]:
    # Keep consistent with tasks/t2i/longtext_bench/score.py
    if mode == "en":
        words_gt = text_gt.split()
        words_ocr = ocr_str.split()
        gt_counter = Counter(words_gt)
        ocr_counter = Counter(words_ocr)
        match = _counter2list(gt_counter & ocr_counter)
        unmatch = _counter2list(gt_counter - ocr_counter)
    else:
        words_gt = text_gt
        gt_counter = Counter(text_gt)
        ocr_counter = Counter(ocr_str)
        match = _counter2list(gt_counter & ocr_counter)
        unmatch = _counter2list(gt_counter - ocr_counter)
    return match, words_gt, unmatch


@EVALUATORS.register_module()
class LongTextBenchEvaluator:
    """Evaluate LongTextBench generations via an HTTP vision-language model."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str],
        prompt: str = (
            "Recognize the text in the image, only reply with the text content. "
            "If no text is recognized, reply with 'No text recognized'."
        ),
        max_workers: int = 8,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_image_size: Optional[int] = None,
        min_short_side: Optional[int] = None,
        max_long_side: Optional[int] = None,
        retry_time: Optional[int] = None,
        use_cache: bool = True,
        cache_chat_name: Optional[str] = None,
        mode: str = "en",
        en_id_max: int = 159,
        **kwargs,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("BAAI_OPENAI_API_KEY") or os.getenv(
            "OPENAI_API_KEY"
        )
        self.base_url = base_url or os.getenv("BAAI_OPENAI_BASE_URL") or os.getenv(
            "OPENAI_BASE_URL"
        )
        self.prompt = prompt
        self.max_workers = max_workers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_image_size = max_image_size
        self.min_short_side = min_short_side
        self.max_long_side = max_long_side
        self.retry_time = retry_time
        self.use_cache = use_cache
        self.cache_chat_name = cache_chat_name
        self.mode = mode
        self.en_id_max = int(en_id_max)

        if not self.api_key:
            logger.warning(
                "LongTextBenchEvaluator: api_key not provided; set OPENAI_API_KEY/BAAI_OPENAI_API_KEY or pass api_key."
            )
        if not self.base_url:
            logger.warning(
                "LongTextBenchEvaluator: base_url not provided; set OPENAI_BASE_URL/BAAI_OPENAI_BASE_URL or pass base_url."
            )
        self._client_kwargs = dict(
            model_name=self.model,
            api_key=self.api_key,
            url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            use_cache=self.use_cache,
            cache_chat_name=self.cache_chat_name,
            stream=False,
            max_image_size=self.max_image_size,
            min_short_side=self.min_short_side,
            max_long_side=self.max_long_side,
            retry_time=self.retry_time,
        )
        self.client = HttpClient(**self._client_kwargs)

    def _build_messages(self, image_path: str) -> List[Dict[str, Any]]:
        image_b64 = encode_image(
            image_path,
            max_size=self.max_image_size,
            min_short_side=self.min_short_side,
            max_long_side=self.max_long_side,
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ]

    def _evaluate_one(
        self, prompt_id: int, prompt: str, gt_text: Any, image_path: str, mode: str
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        messages = self._build_messages(image_path)
        resp = self.client.infer(
            chat_messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not isinstance(resp, ApiResponse):
            raise TypeError(f"Unexpected response type: {type(resp)}")

        pred_text_raw = _clean_text(resp.content or "")
        if isinstance(gt_text, list):
            gt_text_raw = " ".join([str(x) for x in gt_text])
        else:
            gt_text_raw = str(gt_text or "")

        # Score aligned with tasks/t2i/longtext_bench/score.py
        ocr_results = _preprocess_string(pred_text_raw, mode)
        ocr_gt = _preprocess_string(gt_text_raw, mode)
        match, gt_units, _unmatch = _calculate_char_match_ratio(
            ocr_gt, ocr_results, mode
        )
        match_word_count = len(match)
        gt_word_count = len(gt_units) if gt_units is not None else 0
        text_accuray = (match_word_count / gt_word_count) if gt_word_count else 0.0

        full_rec: Dict[str, Any] = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "ocr_gt_raw": gt_text,
            "ocr_results_raw": pred_text_raw,
            "ocr_gt": ocr_gt,
            "ocr_results": ocr_results,
            "image_path": image_path,
            "lang": mode,
        }
        metric_rec: Dict[str, float] = {
            "prompt_id": prompt_id,
            "match_word_count": float(match_word_count),
            "gt_word_count": float(gt_word_count),
            "text_accuray": float(text_accuray),
        }
        return full_rec, metric_rec

    def _lang_by_id(self, prompt_id: int) -> str:
        # Requirement: English ids 0~159, remaining are Chinese.
        return "en" if 0 <= int(prompt_id) <= self.en_id_max else "zh"

    def _resolve_image_path(self, output_dir: str, image_rel: str) -> str:
        return image_rel if osp.isabs(image_rel) else osp.join(output_dir, "samples", image_rel)

    def get_metric_results(
        self,
        output_info: List[Dict[str, Any]],
        output_dir: str,
        annotations: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        tasks: List[Tuple[int, str, Any, str, str]] = []
        for info in output_info:
            qid_raw = info.get("id")
            pid = _safe_int(qid_raw)
            if pid is None:
                continue
            # annotations keys are often zero-padded strings like "0000"
            ann = annotations.get(str(qid_raw), {})
            gt_text = ann.get("text") or ann.get("gt_text") or ""
            prompt = ann.get("prompt") or info.get("prompt", "")
            image_rel = info.get("images") or info.get("image_path")
            if isinstance(image_rel, list):
                image_rel = image_rel[0]
            if not image_rel:
                continue
            image_path = self._resolve_image_path(output_dir, image_rel)
            if not osp.exists(image_path):
                logger.warning(f"Missing image for id={qid_raw}: {image_path}")
                continue
            tasks.append((pid, prompt, gt_text, image_path, self._lang_by_id(pid)))

        if not tasks:
            logger.warning("LongTextBenchEvaluator: nothing to evaluate.")
            return {}

        full_records: Dict[int, Dict[str, Any]] = {}
        metric_records: Dict[int, Dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_to_pid = {
                ex.submit(self._evaluate_one, pid, p, gt, ip, lang): pid
                for pid, p, gt, ip, lang in tasks
            }
            for fut in as_completed(future_to_pid):
                pid = future_to_pid[fut]
                try:
                    full_rec, metric_rec = fut.result()
                except Exception as err:
                    logger.warning(f"LongTextBenchEvaluator failed on id={pid}: {err}")
                    continue
                full_records[pid] = full_rec
                metric_records[pid] = metric_rec

        # attach back to output_info
        for info in output_info:
            pid_raw = info.get("id")
            pid = _safe_int(pid_raw)
            if pid is None or pid not in metric_records:
                continue
            metrics = metric_records[pid]
            rec = full_records[pid]
            info["ocr_gt"] = rec.get("ocr_gt", "")
            info["ocr_results"] = rec.get("ocr_results", "")
            info["ocr_prompt"] = rec.get("prompt", "")
            info["lang"] = rec.get("lang", self._lang_by_id(pid))
            info["match_word_count"] = int(metrics.get("match_word_count", 0.0))
            info["gt_word_count"] = int(metrics.get("gt_word_count", 0.0))
            info["text_accuray"] = float(metrics.get("text_accuray", 0.0))

        match_counts = [
            int(m.get("match_word_count", 0.0)) for m in metric_records.values()
        ]
        gt_counts = [int(m.get("gt_word_count", 0.0)) for m in metric_records.values()]
        results: Dict[str, Any] = {}
        if gt_counts and sum(gt_counts) > 0:
            results["text_score"] = round(sum(match_counts) / sum(gt_counts), 4)
        if match_counts:
            results["match_word_count_sum"] = int(sum(match_counts))
        if gt_counts:
            results["gt_word_count_sum"] = int(sum(gt_counts))

        # Split stats: English ids 0~en_id_max, remaining Chinese.
        en_match_sum = 0
        en_gt_sum = 0
        zh_match_sum = 0
        zh_gt_sum = 0
        for pid, m in metric_records.items():
            lang = self._lang_by_id(pid)
            mwc = int(m.get("match_word_count", 0.0))
            gwc = int(m.get("gt_word_count", 0.0))
            if lang == "en":
                en_match_sum += mwc
                en_gt_sum += gwc
            else:
                zh_match_sum += mwc
                zh_gt_sum += gwc

        if en_gt_sum > 0:
            results["text_score_en"] = round(en_match_sum / en_gt_sum, 4)
        results["match_word_count_sum_en"] = int(en_match_sum)
        results["gt_word_count_sum_en"] = int(en_gt_sum)

        if zh_gt_sum > 0:
            results["text_score_zh"] = round(zh_match_sum / zh_gt_sum, 4)
        results["match_word_count_sum_zh"] = int(zh_match_sum)
        results["gt_word_count_sum_zh"] = int(zh_gt_sum)

        return results
