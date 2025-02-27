register_dataset = {"mlvu_dataset.py": "MLVUDataset"}
register_evaluator = {"mlvu_dataset_evaluator.py": "MLUVDatasetEvaluator"}

dataset = dict(
    type="MLVUDataset",
    data_root="MLVU/MVLU",
    split="dev",
    name="mlvu_dev",
)

evaluator = dict(type="MLUVDatasetEvaluator")
