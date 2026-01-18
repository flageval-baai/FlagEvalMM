# Animal-Bench Task Integration

This task integrates the Animal-Bench dataset into FlagEvalMM framework.

## Files

- `process.py`: Data processing script that converts Animal-Bench JSON files to standard format
- `animalbench_test.py`: Task configuration file

## Data Structure

Animal-Bench includes 20 JSON files covering:
- Common tasks: action_count, action_localization, action_prediction, action_sequence, object_count, object_existence, reasoning
- Animal Kingdom (AK) tasks: action_recognition, object_recognition, bm, pd, pm, sa
- LoTE-Animal tasks: bm, sa
- MammalNet (mmnet) tasks: action_recognition, object_recognition, bm, pd, pm, sa

## Video Path Organization

Videos are expected to be organized under `videos/` with subfolders mapped by
JSON task name:
```
videos/
  ├── TGIF-QA/                       # action_count
  ├── animal_kingdom/
  │   ├── video/                     # AK_* tasks
  │   └── video_grounding/           # action_localization/prediction/sequence
  ├── LoTE-Animal/                   # LoTE_* tasks
  ├── mmnet/                         # mmnet_* and object_existence
  ├── MSRVTT-QA/                     # object_count
  └── NExT-QA/                       # reasoning
```

Mapping used in `process.py`:
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

## Configuration

Before using, update `animalbench_test.py`:
1. Dataset repo ID is set to `"jynkris1016/Animal-Bench"`
2. Ensure video paths match the expected structure above

## Usage

The task follows the same pattern as MVBench:
1. Data is downloaded from HuggingFace (or uses local data if available)
2. JSON files are processed and converted to standard format
3. Video paths are constructed based on sub-task mapping
4. Final `data.json` is saved in `{processed_dataset_path}/{split}/data.json`

## Notes

- The download function will skip if HuggingFace download fails (for local development)
- If JSON directory doesn't exist after download, it will try to use local `Animal-Bench/data` directory
- Video paths are relative to `data_root` which is `{cache_dir}/{processed_dataset_path}/{split}`
- The HuggingFace repo stores videos as zip files under `videos/`; extract them so the folder layout matches the mapping above