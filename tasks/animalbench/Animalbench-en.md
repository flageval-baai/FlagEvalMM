# AnimalBench Dataset (FlagEvalMM Integration)

This document follows the structure of FlagEvalMM's video QA task introduction and provides an English overview and integration notes for AnimalBench.

## Overview

AnimalBench is a multi-task video question answering dataset focused on animal behavior understanding and reasoning. Each sample includes a video, a question, answer options, and a ground-truth answer.

## Tasks and Sources

AnimalBench provides 20 JSON annotation files covering:

- Common tasks: `action_count`, `action_localization`, `action_prediction`, `action_sequence`, `object_count`, `object_existence`, `reasoning`
- Animal Kingdom (AK): `AK_action_recognition`, `AK_object_recognition`, `AK_bm`, `AK_pd`, `AK_pm`, `AK_sa`
- LoTE-Animal: `LoTE_bm`, `LoTE_sa`
- MammalNet (mmnet): `mmnet_action_recognition`, `mmnet_object_recognition`, `mmnet_bm`, `mmnet_pd`, `mmnet_pm`, `mmnet_sa`

## Annotation Format (JSON)

Each JSON file is a list of samples with fields such as:

- `question`: question text
- `candidates`: list of candidate answers
- `answer`: ground-truth answer (text)
- `video`: video filename

After processing, `data.json` is generated with fields:

- `question`, `options`, `answer`, `question_type`, `video_path`, `sub_task`, `question_id`

## Video Directory Layout

FlagEvalMM treats `video_path` as relative to `data_root`. AnimalBench videos should be organized as follows (per `process.py` mapping):

```
videos/
  ├── TGIF-QA/
  ├── animal_kingdom/
  │   ├── video/
  │   └── video_grounding/
  ├── LoTE-Animal/
  ├── mmnet/
  ├── MSRVTT-QA/
  └── NExT-QA/
```

JSON to video directory mapping:

```
action_count -> TGIF-QA
action_localization -> animal_kingdom/video_grounding
action_prediction -> animal_kingdom/video_grounding
action_sequence -> animal_kingdom/video_grounding
AK_action_recognition -> animal_kingdom/video
AK_bm -> animal_kingdom/video
AK_object_recognition -> animal_kingdom/video
AK_pd -> animal_kingdom/video
AK_pm -> animal_kingdom/video
AK_sa -> animal_kingdom/video
LoTE_bm -> LoTE-Animal
LoTE_sa -> LoTE-Animal
mmnet_action_recognition -> mmnet
mmnet_bm -> mmnet
mmnet_object_recognition -> mmnet
mmnet_pd -> mmnet
mmnet_pm -> mmnet
mmnet_sa -> mmnet
object_count -> MSRVTT-QA
object_existence -> mmnet
reasoning -> NExT-QA
```

## Integration in FlagEvalMM

- Config: `tasks/animalbench/animalbench_test.py`
- Processor: `tasks/animalbench/process.py`
- Dataset repo: `jynkris1016/Animal-Bench`
- Output: `{cache_dir}/AnimalBench/test/data.json`

The processing flow mirrors MVBench: download (or use local JSON), build video paths, then write `data.json` in the standard format.

## Usage Notes

1. The HuggingFace repo stores videos as zip files under `videos/`; extract them to match the directory layout above.
2. If download fails, it falls back to local `FlagEvalMM/Animal-Bench/data` for JSON.
3. Run `flagevalmm.dataset.data_preprocessor` to generate the processed dataset.
