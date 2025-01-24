from torch.utils.data import Dataset

from flagevalmm.server.utils import get_meta, get_data


class ServerDataset(Dataset):
    """
    Get data from the server
    """

    def __init__(
        self,
        task_name: str,
        server_ip: str = "http://localhost",
        server_port: int = 5000,
        timeout: int = 1000,
        task_type: str = "vqa",
    ) -> None:
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.task_name = task_name
        self.task_type = task_type
        meta_info = get_meta(task_name, self.server_ip, self.server_port)
        self.datasetname = meta_info["name"]
        self.length: int = meta_info["length"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        data = self.get_data(index)
        question_id = data["question_id"]
        qs = data["question"]
        if self.task_type == "video_qa":
            data_path = data["video_path"]
            multi_modal_data = {"video": data_path}
        else:
            data_path = data["img_path"]
            multi_modal_data = {"image": data_path}
        return question_id, multi_modal_data, qs

    def get_data(self, index: int):
        data = get_data(index, self.task_name, self.server_ip, self.server_port)
        return data
