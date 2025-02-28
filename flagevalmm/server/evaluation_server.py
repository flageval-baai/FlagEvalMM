import os
import os.path as osp
from typing import Any, Dict
from collections import OrderedDict
from flask import Flask, jsonify, request
from flagevalmm.registry import EVALUATORS, DATASETS
from flagevalmm.common.logger import get_logger
import logging
import multiprocessing
import time

# multiprocessing.set_start_method("spawn", force=True)

logger = get_logger(__name__)


class EvaluationServer:
    def __init__(
        self,
        config_dict: Dict,
        model_path: str,
        output_dir: str,
        port: int = 5000,
        host: str = "127.0.0.1",
        debug: bool = True,
        quiet: bool = False,
    ) -> None:
        self.debug = debug
        self.port = port
        self.host = host
        self.config_dict = config_dict
        self.model_path = model_path
        self.output_dir = output_dir
        self.active_task: OrderedDict[str, Any] = OrderedDict()
        self.max_active_task_num = 10
        self.quiet = quiet
        self.app = Flask(__name__)
        self.active_processes: Dict[str, multiprocessing.Process] = {}

        if self.quiet:
            # Disable Flask's default logging
            self.app.logger.disabled = True
            log = logging.getLogger("werkzeug")
            log.disabled = True

        self.set_routes()

    def load_dataset(self, task_name: str) -> None:
        logger.info(f"Loading dataset for task {task_name}")
        self.active_task[task_name] = DATASETS.build(
            self.config_dict[task_name].dataset
        )
        logger.info(f"Dataset for task {task_name} loaded")
        if len(self.active_task) > self.max_active_task_num:
            # pop front
            self.active_task.popitem(last=False)

    def get_task_meta_info(self, task_name: str) -> Dict:
        if task_name not in self.active_task:
            self.load_dataset(task_name)
        meta_info: Dict = self.active_task[task_name].meta_info()
        meta_info["output_dir"] = osp.join(self.output_dir, task_name)
        os.makedirs(meta_info["output_dir"], exist_ok=True)
        return meta_info

    def get_task_data_by_index(self, task_name: str, index: int):
        if task_name not in self.active_task:
            self.load_dataset(task_name)
        return self.active_task[task_name][index]

    def evaluate_task(
        self, task_name: str, model_name: str, new_output_dir: str = ""
    ) -> None:
        # Create a unique process ID using timestamp
        process_id = f"{task_name}_{model_name}_{int(time.time() * 1000)}"
        # If evaluator is not specified, skip evaluation
        if not self.config_dict[task_name].get("evaluator", None):
            logger.warning(
                f"No evaluator specified for task {task_name}, skipping evaluation"
            )
            return
        evaluator = EVALUATORS.build(self.config_dict[task_name].evaluator)
        start_method = self.config_dict[task_name].evaluator.get("start_method", "fork")
        if task_name not in self.active_task:
            self.load_dataset(task_name)
        dataset = self.active_task[task_name]
        logger.info(f"Starting evaluation process for task {task_name}")
        output_dir = (
            new_output_dir if new_output_dir else osp.join(self.output_dir, task_name)
        )

        # Get the specific context for this process
        ctx = multiprocessing.get_context(start_method)

        # Create and start the process using the context
        process = ctx.Process(
            target=evaluator.process,
            args=(dataset,),
            kwargs={"output_dir": output_dir, "model_name": model_name},
        )
        process.start()
        self.active_processes[process_id] = process

    def shutdown(self):
        # Terminate all running processes
        for process in self.active_processes.values():
            if process.is_alive():
                process.terminate()
                process.join()
        self.active_processes.clear()

    def __del__(self):
        self.shutdown()

    def set_routes(self) -> None:
        @self.app.route("/task_info", methods=["GET"])
        def get_task_info():
            task_info = {
                "task_names": list(self.config_dict.keys()),
                "model_path": self.model_path,
            }
            logger.info(f"Task info: {task_info}")
            return jsonify(task_info)

        @self.app.route("/meta_info", methods=["GET"])
        def get_meta_info():
            task_name = request.args.get("task")
            if task_name not in self.config_dict:
                return f"Task {task_name} not found", 404
            return jsonify(self.get_task_meta_info(task_name))

        @self.app.route("/get_data", methods=["GET"])
        def get_data():
            index = request.args.get("index", type=int)
            task_name = request.args.get("task")
            try:
                data = self.get_task_data_by_index(task_name=task_name, index=index)
            except IndexError:
                return "Index out of range", 400
            except Exception as e:
                err_msg = f"Error Type: {type(e).__name__} Error: {str(e)}"
                logger.error(err_msg)
                return err_msg, 500
            return jsonify(data)

        @self.app.route("/evaluate", methods=["GET"])
        def evaluate():
            task_name = request.args.get("task")
            model_name = request.args.get("model_name", "unnamed")
            new_output_dir = request.args.get("output_dir", "")

            if task_name not in self.config_dict:
                return f"Task {task_name} not found", 404

            self.evaluate_task(task_name, model_name, new_output_dir)
            return jsonify("Evaluation process started")

        @self.app.route("/get_retrieval_data", methods=["GET"])
        def get_retrieval_data():
            index = request.args.get("index", type=int)
            data_type = request.args.get("type", type=str)
            task_name = request.args.get("task")
            if data_type not in ["img", "text"]:
                return "Invalid data type", 400
            try:
                if task_name not in self.active_task:
                    self.load_dataset(task_name)
                data = self.active_task[task_name].get_data(index, data_type)
            except IndexError:
                return "Index out of range", 400
            except Exception as e:
                return str(e), 500
            return jsonify(data)

        @self.app.route("/eval_finished", methods=["GET"])
        def eval_finished():
            RUNNING = 0
            FINISHED = 1

            # Clean up finished processes
            for process_id, process in list(self.active_processes.items()):
                if not process.is_alive():
                    process.join()
                    del self.active_processes[process_id]

            # If there are any active processes, evaluation is still running
            if self.active_processes:
                return jsonify({"status": RUNNING})
            return jsonify({"status": FINISHED})

    def run(self) -> None:
        self.app.run(debug=self.debug, port=self.port, host=self.host)

    def get_flask_app(self) -> Flask:
        return self.app
