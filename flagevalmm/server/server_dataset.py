from torch.utils.data import Dataset


class ServerDataset(Dataset):
    """
    Get data from the server
    """

    def __init__(
        self,
        task_name: str,
        task_manager,
        task_type: str = "vqa",
    ) -> None:
        # from flagevalmm.models.base_model_adapter import TaskManager
        self.task_manager = task_manager
        self.task_name = task_name
        self.task_type = task_type
        meta_info = self.task_manager.get_meta_info(task_name)
        self.datasetname = meta_info["name"]
        self.length: int = meta_info["length"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        data = self.get_data(index)
        question_id = data["question_id"]
        qs = data["question"]
        if data.get("video_path", None):
            data_path = data["video_path"]
            multi_modal_data = {"video": data_path}
        else:
            data_path = data["img_path"]
            multi_modal_data = {"image": data_path}
        return question_id, multi_modal_data, qs

    def get_data(self, index: int):
        data = self.task_manager.get_data(self.task_name, index)
        return data
