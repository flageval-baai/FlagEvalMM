import contextlib
import io
import os
import os.path as osp
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from flagevalmm.common.logger import get_logger
from flagevalmm.models.api_response import ApiResponse
from flagevalmm.models.http_client import HttpClient
from flagevalmm.prompt.prompt_tools import encode_image
from flagevalmm.registry import EVALUATORS

logger = get_logger(__name__)


WISE_SYSTEM_PROMPT = (
    "You are a professional Vincennes image quality audit expert, "
    "please evaluate the image quality strictly according to the protocol."
)


def _build_user_prompt(prompt: str, explanation: str) -> str:
    # Keep this aligned with tasks/t2i/wise/gpt_eval.py (protocol + strict output format).
    return f"""Please evaluate strictly and return ONLY the three scores as requested.

# Text-to-Image Quality Evaluation Protocol

## System Instruction
You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores.

**Input Parameters**
- PROMPT: [User's original prompt to]
- EXPLANATION: [Further explanation of the original prompt]
---

## Scoring Criteria

**Consistency (0-2):**  How accurately and completely the image reflects the PROMPT.
* **0 (Rejected):**  Fails to capture key elements of the prompt, or contradicts the prompt.
* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.  Noticeable deviations from the prompt's intent.
* **2 (Exemplary):**  Perfectly and completely aligns with the PROMPT.  Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

**Realism (0-2):**  How realistically the image is rendered.
* **0 (Rejected):**  Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements.  While somewhat believable, noticeable flaws detract from realism.
* **2 (Exemplary):**  Achieves photorealistic quality, indistinguishable from a real photograph.  Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

**Aesthetic Quality (0-2):**  The overall artistic appeal and visual quality of the image.
* **0 (Rejected):**  Poor aesthetic composition, visually unappealing, and lacks artistic merit.
* **1 (Conditional):**  Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
* **2 (Exemplary):**  Possesses exceptional aesthetic quality, comparable to a masterpiece.  Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

---

## Output Format

**Do not include any other text, explanations, or labels.** You must return only three lines of text, each containing a metric and the corresponding score, for example:

**Example Output:**
Consistency: 2
Realism: 1
Aesthetic Quality: 0

---

**IMPORTANT Enforcement:**

Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

---
Here are the Prompt and EXPLANATION for this evaluation:
PROMPT: "{prompt}"
EXPLANATION: "{explanation}"
Please strictly adhere to the scoring criteria and follow the template format when providing your results."""


def _extract_scores(txt: str) -> Dict[str, float]:
    # Expected keys: consistency, realism, aesthetic_quality
    # Be tolerant to minor format variation (colon variants, bold markers, "/2").
    pat = r"\*{0,2}(Consistency|Realism|Aesthetic\s+Quality)\*{0,2}\s*[:：]?\s*([0-2])(?:\s*/\s*2)?"
    matches = re.findall(pat, txt, re.IGNORECASE)
    out: Dict[str, float] = {}
    for k, v in matches:
        out[k.lower().replace(" ", "_")] = float(v)
    return out


def _calculate_wiscore(consistency: float, realism: float, aesthetic_quality: float) -> float:
    # Same as tasks/t2i/wise/calculate.py
    return (0.7 * consistency + 0.2 * realism + 0.1 * aesthetic_quality) / 2.0


def _prompt_id_to_category(prompt_id: int) -> Optional[str]:
    # Same segment rules as tasks/t2i/wise/calculate.py
    if 1 <= prompt_id <= 400:
        return "CULTURE"
    if 401 <= prompt_id <= 567:
        return "TIME"
    if 568 <= prompt_id <= 700:
        return "SPACE"
    if 701 <= prompt_id <= 800:
        return "BIOLOGY"
    if 801 <= prompt_id <= 900:
        return "PHYSICS"
    if 901 <= prompt_id <= 1000:
        return "CHEMISTRY"
    return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


@EVALUATORS.register_module()
class WiseEvaluator:
    """
    WiSE evaluator:
    - Uses a vision-capable LLM to score each generated image on:
      Consistency / Realism / Aesthetic Quality in {0,1,2}
    - Computes WiScore per sample and aggregated category/overall scores.

    The scoring prompt and category aggregation logic are adapted from:
    - tasks/t2i/wise/gpt_eval.py
    - tasks/t2i/wise/calculate.py
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-05-13",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_workers: int = 10,
        max_image_size: Optional[int] = None,
        min_short_side: Optional[int] = None,
        max_long_side: Optional[int] = None,
        validate_prompt_ids: bool = False,
        category: str = "all",
        use_cache: bool = True,
        cache_chat_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("BAAI_OPENAI_API_KEY") or os.getenv(
            "OPENAI_API_KEY"
        )
        self.base_url = base_url or os.getenv("BAAI_OPENAI_BASE_URL") or os.getenv(
            "OPENAI_BASE_URL"
        )
        self.max_workers = max_workers
        self.max_image_size = max_image_size
        self.min_short_side = min_short_side
        self.max_long_side = max_long_side
        self.validate_prompt_ids = validate_prompt_ids
        self.category = category
        self.use_cache = use_cache
        self.cache_chat_name = cache_chat_name
        self.retry_time = kwargs.get("retry_time")

        if not self.api_key:
            logger.warning(
                "WiseEvaluator: api_key not provided; set OPENAI_API_KEY/BAAI_OPENAI_API_KEY "
                "or pass api_key in config."
            )
        if not self.base_url:
            logger.warning(
                "WiseEvaluator: base_url not provided; set OPENAI_BASE_URL/BAAI_OPENAI_BASE_URL "
                "or pass base_url in config."
            )
        self._client_kwargs = dict(
            model_name=self.model,
            api_key=self.api_key,
            url=self.base_url,
            use_cache=self.use_cache,
            cache_chat_name=self.cache_chat_name,
            stream=False,
            max_image_size=self.max_image_size,
            min_short_side=self.min_short_side,
            max_long_side=self.max_long_side,
            retry_time=self.retry_time,
        )
        self.client = HttpClient(**self._client_kwargs)

    def _build_messages(self, prompt: str, explanation: str, image_b64: str) -> List[Dict]:
        user_text = _build_user_prompt(prompt=prompt, explanation=explanation)
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": WISE_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ]

    def _evaluate_one(
        self,
        prompt_id: int,
        prompt: str,
        explanation: str,
        image_path: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        image_b64 = encode_image(
            image_path,
            max_size=self.max_image_size,
            min_short_side=self.min_short_side,
            max_long_side=self.max_long_side,
        )
        messages = self._build_messages(
            prompt=prompt, explanation=explanation, image_b64=image_b64
        )
        # BaseApiModel._single_infer prints streamed chunks; silence it here.

        resp = self.client.infer(
            chat_messages=messages, temperature=0.0, max_tokens=4096
        )
        assert isinstance(resp, ApiResponse), f"Unexpected response type: {type(resp)}"
        eval_txt = resp.content or ""
        scores = _extract_scores(eval_txt)
        consistency = float(scores.get("consistency", 0.0))
        realism = float(scores.get("realism", 0.0))
        aesthetic_quality = float(scores.get("aesthetic_quality", 0.0))
        wiscore = _calculate_wiscore(consistency, realism, aesthetic_quality)
        full_rec = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "explanation": explanation,
            "image_path": image_path,
            "evaluation": eval_txt,
        }
        score_rec = {
            "prompt_id": prompt_id,
            "consistency": consistency,
            "realism": realism,
            "aesthetic_quality": aesthetic_quality,
            "wiscore": wiscore,
            "category": _prompt_id_to_category(prompt_id),
        }
        return full_rec, score_rec

    def _expected_prompt_ids(self) -> Optional[set[int]]:
        if self.category == "culture":
            return set(range(1, 401))
        if self.category == "space-time":
            return set(range(401, 701))
        if self.category == "science":
            return set(range(701, 1001))
        if self.category == "all":
            return set(range(1, 1001))
        return None

    def get_metric_results(
        self, output_info: List[Dict], output_dir: str, annotations: Dict, **kwargs
    ) -> Dict[str, Any]:
        tasks: List[Tuple[int, str, str, str]] = []
        present_prompt_ids: set[int] = set()

        for info in output_info:
            qid = info.get("id")
            prompt_id = _safe_int(qid)
            if prompt_id is None:
                raise ValueError(f"WiseEvaluator: invalid prompt_id: {qid}")
            present_prompt_ids.add(prompt_id)

            ann = annotations.get(str(qid), {})
            prompt = ann.get("Prompt", ann.get("prompt", info.get("prompt", "")))
            explanation = ann.get("Explanation", ann.get("explanation", ""))

            image_rel = info.get("images")
            if isinstance(image_rel, list):
                image_rel = image_rel[0]
            if not image_rel:
                continue
            image_path = osp.join(output_dir, "samples", image_rel)
            
            if not osp.exists(image_path):
                logger.warning(f"Missing image: {image_path}")
                continue
            tasks.append((prompt_id, str(prompt), str(explanation), image_path))

        if tasks:
            logger.info(
                f"WiseEvaluator: evaluating {len(tasks)} images (use_cache={self.use_cache})"
            )
        else:
            logger.info("WiseEvaluator: nothing to evaluate (no valid items)")

        # Multi-threaded evaluation
        exist_full: Dict[int, Dict[str, Any]] = {}
        exist_scores: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_to_pid = {
                ex.submit(self._evaluate_one, pid, p, e, ip): pid for pid, p, e, ip in tasks
            }
            for fut in as_completed(future_to_pid):
                pid = future_to_pid[fut]
                try:
                    full_rec, score_rec = fut.result()
                except Exception as err:
                    logger.warning(f"WiseEvaluator: failed on prompt_id={pid}: {err}")
                    continue
                exist_full[pid] = full_rec
                exist_scores[pid] = score_rec

        # Optional prompt_id validation (aligned with tasks/t2i/wise/calculate.py)
        if self.validate_prompt_ids:
            expected = self._expected_prompt_ids()
            if expected is not None:
                missing = expected - present_prompt_ids
                if missing:
                    raise ValueError(
                        f"WiseEvaluator: missing prompt_ids for category={self.category}: "
                        f"{sorted(list(missing))}"
                    )

        score_sorted = [exist_scores[k] for k in sorted(exist_scores.keys())]
        full_by_id: Dict[int, Dict[str, Any]] = {
            int(k): v for k, v in exist_full.items() if _safe_int(k) is not None
        }

        # Attach per-sample results back to output_info
        scores_by_id = {int(r["prompt_id"]): r for r in score_sorted if _safe_int(r.get("prompt_id")) is not None}
        for info in output_info:
            pid = _safe_int(info.get("id"))
            if pid is None or pid not in scores_by_id:
                continue
            r = scores_by_id[pid]
            info["wise_consistency"] = r.get("consistency", 0.0)
            info["wise_realism"] = r.get("realism", 0.0)
            info["wise_aesthetic_quality"] = r.get("aesthetic_quality", 0.0)
            info["wiscore"] = r.get("wiscore", 0.0)
            info["wise_category"] = r.get("category")
            # Save raw evaluator text alongside scores (requested: save eval_txt together).
            full_rec = full_by_id.get(pid, {})
            info["wise_eval_txt"] = full_rec.get("evaluation", "")

        # Aggregate results (aligned with calculate.py)
        metric_scores = defaultdict(list)
        for r in score_sorted:
            pid = _safe_int(r.get("prompt_id"))
            if pid is None:
                continue
            metric_scores["wiscore_mean"].append(float(r.get("wiscore", 0.0)))
            cat = _prompt_id_to_category(pid)
            if cat is not None:
                metric_scores[f"wiscore_{cat}"].append(float(r.get("wiscore", 0.0)))

        results: Dict[str, Any] = {}
        if metric_scores["wiscore_mean"]:
            results["wiscore_mean"] = round(
                sum(metric_scores["wiscore_mean"]) / len(metric_scores["wiscore_mean"]), 4
            )

        ordered_categories = ["CULTURE", "TIME", "SPACE", "BIOLOGY", "PHYSICS", "CHEMISTRY"]
        for cat in ordered_categories:
            key = f"wiscore_{cat}"
            vals = metric_scores.get(key, [])
            if vals:
                results[key] = round(sum(vals) / len(vals), 4)
                results[f"{key}_num"] = len(vals)

        # Overall WiScore across all categories (same weights as calculate.py)
        if all(f"wiscore_{cat}" in results for cat in ordered_categories):
            results["wiscore_overall"] = round(
                0.4 * results["wiscore_CULTURE"]
                + 0.167 * results["wiscore_TIME"]
                + 0.133 * results["wiscore_SPACE"]
                + 0.1 * results["wiscore_BIOLOGY"]
                + 0.1 * results["wiscore_PHYSICS"]
                + 0.1 * results["wiscore_CHEMISTRY"],
                4,
            )

        return results

