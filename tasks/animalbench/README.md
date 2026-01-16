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

Videos are expected to be organized as:
```
videos/
  ├── AK/          # Animal Kingdom videos
  ├── LoTE/        # LoTE-Animal videos
  ├── mmnet/       # MammalNet videos
  └── *.mp4        # Common task videos
```

## Configuration

Before using, update `animalbench_test.py`:
1. Replace `"YourOrg/Animal-Bench"` with actual HuggingFace repo ID
2. Ensure video paths match the expected structure

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
