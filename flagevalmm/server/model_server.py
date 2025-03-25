import atexit
import threading
import subprocess
import time
import requests
import shlex
from typing import List, Optional
from flagevalmm.common.logger import get_logger
import os

os.environ["no_proxy"] = "127.0.0.1,localhost"

logger = get_logger(__name__)


class ModelServer:
    """
    Currently, it is only used for vllm server, it will be support SGLang etc. in the future.
    """

    def __init__(
        self,
        model_name: str,
        port: int = 8000,
        server_type: str = "vllm",
        extra_args: Optional[str] = None,
    ):
        self.model_name = model_name
        self.port = port
        assert server_type in ["vllm", "sglang"], "server_type must be vllm or sglang"
        # extra args is like "--limit-mm-per-prompt image=8 --max-model-len 32768"
        splited_args = shlex.split(extra_args) if extra_args else []
        if server_type == "vllm":
            self.get_cmd = self.get_vllm_cmd
        else:
            self.get_cmd = self.get_sglang_cmd
        self.execute_cmd = None
        self.launch_server(splited_args)

    def get_vllm_cmd(self, args: List):
        cmd = ["vllm", "serve", self.model_name, "--port", str(self.port), *args]
        return cmd

    def get_sglang_cmd(self, args: List):
        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_name,
            "--port",
            str(self.port),
            *args,
        ]
        return cmd

    def launch_server(self, args: List):
        cmd = self.get_cmd(args)
        self.execute_cmd = cmd
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        self.stop_event = threading.Event()
        atexit.register(self.cleanup)

        def log_subprocess_output(pipe, stop_event):
            # Read lines until stop event is set
            for line in iter(pipe.readline, ""):
                if stop_event.is_set():
                    break
                else:
                    print(line, end="")
            pipe.close()
            print("server log tracking thread stopped successfully.")

        self.stdout_thread = threading.Thread(
            target=log_subprocess_output,
            args=(self.server_process.stdout, self.stop_event),
        )
        self.stderr_thread = threading.Thread(
            target=log_subprocess_output,
            args=(self.server_process.stderr, self.stop_event),
        )
        self.stdout_thread.start()
        self.stderr_thread.start()

        server_ready = False
        while not server_ready:
            # Check if the process has terminated unexpectedly
            if self.server_process.poll() is not None:
                # Output the captured logs
                stdout, stderr = self.server_process.communicate()
                print(stdout)
                print(stderr)
                raise Exception(
                    f"Subprocess terminated unexpectedly with code {self.server_process.returncode}"
                )
            try:
                # Make a simple request to check if the server is up
                response = requests.get(f"http://localhost:{self.port}/v1/models")
                if response.status_code == 200:
                    server_ready = True
                    print("server is ready!")
            except requests.exceptions.ConnectionError:
                # If the connection is not ready, wait and try again
                time.sleep(1)
        # close the vllm server output
        self.stop_event.set()

    def stop(self):
        self.cleanup()

    def cleanup(self):
        if hasattr(self, "server_process"):
            self.server_process.terminate()
            logger.info("VLLM server terminated")
            try:
                # Wait for the process to terminate fully
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.info(
                    "VLLM server did not terminate within timeout, forcefully terminating"
                )
                self.server_process.kill()
                self.server_process.wait()

        if hasattr(self, "stop_event"):
            self.stop_event.set()
        if hasattr(self, "stdout_thread"):
            self.stdout_thread.join()
        if hasattr(self, "stderr_thread"):
            self.stderr_thread.join()
