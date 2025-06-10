Add New Tasks
====================

FlagEvalMM supports a wide variety of multimodal evaluation tasks and benchmarks.

You can create custom tasks by implementing dataset and evaluator classes:

.. code-block:: python

   from flagevalmm.dataset import BaseDataset
   from flagevalmm.evaluator import BaseEvaluator
   from flagevalmm.registry import DATASETS, EVALUATORS

   @DATASETS.register_module()
   class CustomDataset(BaseDataset):
       def __init__(self, data_file, **kwargs):
           super().__init__(**kwargs)
           # Implementation

   @EVALUATORS.register_module()
   class CustomEvaluator(BaseEvaluator):
       def evaluate(self, predictions, annotations):
           # Implementation
           return results 