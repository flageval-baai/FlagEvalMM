# 添加自定义评测任务

本指南将介绍如何添加新的评测任务到系统中。主要包含三个步骤：

## 1. 创建评测任务配置

在 `tasks` 目录下创建新的文件夹和对应的配置文件。对于简单的评测任务（如 VQA），可以直接使用 `VqaBaseDataset` 类。

基本配置模板如下：

```python
# tasks/custom_task/custom_task.py

config = dict(
    dataset_path="<数据集路径>",  # 原始数据集路径
    split="image",               # 数据集分割
    processed_dataset_path="CustomBench",  # 处理后数据集保存路径
    processor="process.py",      # 数据处理脚本
)

# 配置方式 1: 使用默认 prompt
dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(type="PromptTemplate"),  # 默认 prompt: 'Answer the question using a single word or phrase.'
    name="custom_bench",
)

# 配置方式 2: 自定义 prompt
dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate", 
        pre_prompt="",   # prompt 前缀
        post_prompt=""   # prompt 后缀
    ),
    name="custom_bench",
)

evaluator = dict(
    type="BaseEvaluator", 
    eval_func="evaluate.py"
)
```

## 2. 实现评估逻辑

创建 `evaluate.py` 文件并实现评估函数。该函数需要返回包含评估指标的字典。

```python
# tasks/custom_task/evaluate.py

def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """评估模型预测结果
    
    Args:
        annotations: 标注数据
        predictions: 模型预测结果
        
    Returns:
        Dict: 包含评估指标的字典
    """
    results = {}
    results["question_acc"] = cal_accuracy(annotations, predictions)
    return results
```

## 3. 实现数据处理逻辑

创建 `process.py` 文件并实现数据处理函数。该函数需要将原始数据转换为标准格式。

```python
# tasks/custom_task/process.py

def process(cfg):
    """处理原始数据集
    
    处理后的数据必须包含以下字段：
    - question_id: 问题唯一标识
    - img_path: 图片路径
    - question: 问题文本
    - question_type: 问题类型
    """
    # 加载原始数据
    dataset = load_dataset(cfg)
    
    # 设置输出路径
    output_dir = osp.join(cfg.processed_dataset_path, cfg.split, cfg.get("dataset_name", ""))
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理数据
    processed_data = []
    for i, item in enumerate(tqdm.tqdm(dataset)):
        processed_item = {
            "question_id": f'{item["data_type"]}_{str(i)}',
            "img_path": item["image_path"],
            "question": item["question"],
            "question_type": item["data_type"]
        }
        processed_data.append(processed_item)
    
    # 保存处理后的数据
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(processed_data, f)
```
