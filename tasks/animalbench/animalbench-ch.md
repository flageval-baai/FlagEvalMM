# AnimalBench 数据集（FlagEvalMM 集成说明）

本页面参考 FlagEvalMM 视频问答任务介绍文档的结构，提供 AnimalBench 的中文概览与集成说明。

## 数据集概览

AnimalBench 是面向动物行为理解与推理的多任务视频问答数据集，包含多种子任务与来源数据子集。每条样本提供视频、问题、候选答案以及标准答案。

## 子任务与来源

AnimalBench 共有 20 个 JSON 标注文件，涵盖以下任务类型：

- Common tasks: `action_count`, `action_localization`, `action_prediction`, `action_sequence`, `object_count`, `object_existence`, `reasoning`
- Animal Kingdom (AK): `AK_action_recognition`, `AK_object_recognition`, `AK_bm`, `AK_pd`, `AK_pm`, `AK_sa`
- LoTE-Animal: `LoTE_bm`, `LoTE_sa`
- MammalNet (mmnet): `mmnet_action_recognition`, `mmnet_object_recognition`, `mmnet_bm`, `mmnet_pd`, `mmnet_pm`, `mmnet_sa`

## 标注格式（JSON）

每个 JSON 文件是一个样本列表，核心字段包含：

- `question`: 问题文本
- `candidates`: 候选答案列表
- `answer`: 正确答案（文本）
- `video`: 视频文件名

处理后会生成 `data.json`，字段包括：

- `question`, `options`, `answer`, `question_type`, `video_path`, `sub_task`, `question_id`

## 视频目录结构

FlagEvalMM 会将 `video_path` 视为相对于 `data_root` 的路径。AnimalBench 的视频应按如下目录组织（对应 `process.py` 中的映射）：

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

JSON 与视频目录的对应关系：

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

## 与 FlagEvalMM 的集成方式

- 配置文件：`tasks/animalbench/animalbench_test.py`
- 处理脚本：`tasks/animalbench/process.py`
- 数据集仓库：`jynkris1016/Animal-Bench`
- 处理输出：`{cache_dir}/AnimalBench/test/data.json`

与 MVBench 的处理流程一致：下载或使用本地 JSON，构造视频路径并生成标准格式 `data.json`。

## 使用提示

1. 从 HuggingFace 下载数据后，确保 `videos/` 下的 zip 已解压为上述目录结构。
2. 若无法下载，会尝试读取本地 `FlagEvalMM/Animal-Bench/data` 的 JSON。
3. 运行 `flagevalmm.dataset.data_preprocessor` 可生成标准格式数据。
