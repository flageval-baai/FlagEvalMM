# README

Please download the dataset following the official instructions: <https://huggingface.co/datasets/OpenGVLab/MVBench>

## Evaluation Results

| Task                      | Score       |
|---------------------------|-------------|
| object_interaction        | 33.0        |
| state_change              | 41.5        |
| fine_grained_pose         | 25.5        |
| action_prediction         | 34.0        |
| scene_transition          | 49.0        |
| counterfactual_inference  | 42.5        |
| action_antonym            | 71.5        |
| fine_grained_action       | 33.0        |
| object_existence          | 48.5        |
| moving_count              | 38.0        |
| moving_direction          | 34.5        |
| action_sequence           | 28.0        |
| object_shuffle            | 32.5        |
| unexpected_action         | 58.0        |
| moving_attribute          | 29.0        |
| episodic_reasoning        | 37.5        |
| character_order           | 42.0        |
| action_count              | 50.5        |
| egocentric_navigation     | 34.5        |
| action_localization       | 38.0        |
| **accuracy**              | **40.05**   |

## Run Command

```bash
flagevalmm \
  --tasks tasks/mvbench/mvbench_test.py \
  --exec model_zoo/vlm/qwen_vl/model_adapter.py \
  --output-dir ./results/Qwen2.5-VL-3B \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --cfg model_configs/open/Qwen2.5-VL-3B-Instruct.json
```
