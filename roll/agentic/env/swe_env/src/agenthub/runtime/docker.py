import mimetypes
import os, sys
import json
from time import sleep
import time
from turtle import setup
import uuid
import tempfile
# import docker
import requests
# from docker.models.containers import Container

from roll.agentic.env.swe_env.src.repo_analysis.execution_log_parser import parse_log_fn, decolor_dict_keys
from roll.agentic.env.swe_env.src.agenthub.runtime.base import (
    ExecutionEnvironment,
)
import base64
import subprocess
import datetime
import hashlib
import shutil

# import docker
import tarfile
import io
import os
from roll.agentic.env.swe_env.src.agenthub.utils.log import get_logger
import re
from roll.agentic.env.swe_env.src.agenthub.utils.utils import match_dockerimage_to_repo
from roll.agentic.env.swe_env.src.agenthub import SUPPORTED_REPOS, SKIP_FILES, SKIP_FILES_NEW, CMD_TIMEOUT
import concurrent.futures

from roll.agentic.env.swe_env.src.get_requirment.swebench_test_spec import make_test_spec
from roll.agentic.env.swe_env.src.agenthub.utils.utils import get_logger
from roll.agentic.env.swe_env.src.commit_models.diff_classes import ParsedCommit
import asyncio
import aiohttp

DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.test_spec.test_spec import TestSpec
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER, get_eval_type
from swebench.harness.grading import get_eval_tests_report, get_resolution_status


##############################################################################
# Docker runtime
##############################################################################
class DockerRuntime(ExecutionEnvironment):
    """
    docker runtime is responsible for the interacting with the docker environment.
    In particular, it should allow for accomodating the features of the particualr docker envs used for r2e-edits
    - collect files
    - list files excluding test files etc
    """

    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str | list[str] = "/bin/bash",
        logger=None,
        route_key: str = None,
        swe_rex_host: str = "https://xrl-aliyun.alibaba-inc.com/swe-rex/docker",
        **docker_kwargs,
    ):
        # check if ds is provided (required for all dockers moving forward)
        assert ds, f"Dataset not provided for docker image: {docker_image}"

        # swebench specific setup
        self.ds = ds
        self.docker_image = (
            self.ds["docker_image"] if not docker_image else docker_image
        )
        self.swebench_verified = "swebench" in self.docker_image
        if self.swebench_verified:
            # also create a test spec for swebench verified dockers (useful for grading)
            self.test_spec = make_test_spec(self.ds)

        # set runtime params
        self.repo_path = repo_path
        self.alt_path = alt_path
        self.command = command
        self.repo_name = (
            self.ds["repo"] if self.swebench_verified else self.ds["repo_name"]
        )
        self.commit_json = (
            self.ds["parsed_commit"]
            if self.swebench_verified
            else self.ds["parsed_commit_content"]
        )
        self.commit = ParsedCommit(**json.loads(self.commit_json))
        self.docker_kwargs = docker_kwargs
        if logger is None:
            self.logger = get_logger(
                "DockerRuntime"
            )  # Pass the module name for clarity
        else:
            self.logger = logger

        # swe-rex
        if route_key is None:
            self.route_key = uuid.uuid4().hex
        else:
            self.route_key = route_key
        # print(f'[runtime.docker]specified route_key: {self.route_key}')
        self.host = swe_rex_host
        # print(f'[runtime.docker]host: {self.host}')

    def reset_container(self):
        self.container_name = self.start_container()
        self.route_key = self.route_key
        # Initialize the environment
        if self.container_name != 'ERROR':
            setup_env_result = self.setup_env() #  "SUCCESS", "ERROR"
            # self.logger.info("Docker environment initialized")
            # self.logger.info("repo name: %s", self.repo_name)
            # self.logger.info("Docker image: %s", self.docker_image)
            # self.logger.info("Container name: %s", self.container_name)
        else:
            setup_env_result = "ERROR"
        return {'container_name': self.container_name, 'setup_env_result': setup_env_result,"reset_retry_times": self.retry_times, "route_key": self.route_key}

    @staticmethod
    def _get_container_name(image_name: str) -> str:
        """Return name of container"""
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-")
        image_name_sanitized = image_name_sanitized.replace(":", "-")
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    def start_container(self):
        """请求远程服务init_env, pull docker image, 并返回container_name。
        @input: 
            - docker_image: 
            - clear_time: 
        @output:
            - route_key: 
            - container_name: 
            - retry_times: 
            - self.session: 
        """
        self.retry_times = 0
        
        max_execute_time,max_execute_retry = 300.0,10 # 最多5min,最多10次
        st,execute_time = time.time(),0
        clear_time = 60 # TODO
        timeout = 180
        error_message = ""

        self.route_key = uuid.uuid4().hex
        self.session = requests.Session()
        self.session.headers.update({
            "ROUTE-KEY": self.route_key
        })

        while execute_time < max_execute_time and self.retry_times < max_execute_retry:
            try:
                response = self.session.post(
                    f'{self.host}/init_env',
                    json={"image": self.docker_image, "auto_clear_time": clear_time},
                    timeout = (10, timeout)
                )
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("status") == "Success":
                        return response_data.get("result").get("container_name")
            except Exception as e: error_message = repr(e)
            
            self.retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)

            if self.retry_times % 11 == 0: # 重更新一次route_key
                self.route_key = uuid.uuid4().hex
                self.session = requests.Session()
                self.session.headers.update({
                    "ROUTE-KEY": self.route_key
                })
        print(
            f"[ERROR][START CONTAINER ERROR](retry_times:{self.retry_times})(execute_time: {execute_time})"
            f"route_key: {self.route_key}, docker_image: {self.docker_image}, timeout: {timeout}, error: {error_message}"
        )
        return 'ERROR'

    async def stop_container(self):
        try:
            if self.container_name:
                asyncio.create_task(self._stop_container_async())
        except Exception as e:
            print(f"[ERROR][{os.getpid()}][STOP CONTAINNER ERROR] route_key: {self.route_key}, container_name: {self.container_name}, error message: {repr(e)}")

    async def _stop_container_async(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{self.host}/stop',
                headers={"ROUTE-KEY": self.route_key},
                json={"container_name": self.container_name}
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    if response_data.get("status") == "Success":
                        self.logger.info(f"Container {self.container_name} stopped successfully")
                    else:
                        self.logger.warning(f"Container stop failed: {response_data}")
                else:
                    print(f"Container stop request failed with status: {response.status}")

    def execute(self, command: str | list[str], workdir: str, environment: dict, timeout=180) -> str:
        """
        @执行成功的返回
            {
                'stdout': "Error executing command:\n\n[STDOUT]\n\n \n\n[STDERR]\n\npython: can't open file '/testbed/reproduce_issue.py': [Errno 2] No such file or directory\n", 
                'stderr': '', 
                'exit_code': 2
            }
        @执行失败的返回
            ERROR: xxx
        """
        data = {
            "command": command,
            "cwd": workdir,
            "env": environment,
            "container_name": self.container_name,
            "timeout": timeout
        }

        max_execute_time,max_execute_retry = 300.0,10 # 最多5min,最多10次
        st,execute_time,retry_times = time.time(),0,0
        error_message = ""
        # self.logger.info(f'[START EXCUTE]execute command: {command}, container_name: {self.container_name}, timeout: {timeout}')
        
        while execute_time < max_execute_time and retry_times < max_execute_retry:
            try:
                response = self.session.post(
                    f"{self.host}/execute",
                    json=data,
                    timeout=timeout
                )
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("status") == "Success":
                        # self.logger.info(f'[EXCUTE SUCCESS](RETRY_times:{retry_times}) execute command: {command}, container_name: {self.container_name}, timeout: {timeout}')
                        return response_data.get("result")
            except Exception as e:
                error_message = repr(e)
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
        print(
            f"[ERROR][EXECUTE ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {self.route_key}, input: {[data]}, timeout: {timeout}, error: {error_message}"
        )
        return "ERROR: " + error_message

    def copy_to_container(self, src_path: str, dest_path: str):
        """
        Copies a file or directory from the host to the Docker container.

        Args:
            src_path: Path to the file or directory on the host.
            dest_path: Destination path inside the container.
        """
        content_type, _ = mimetypes.guess_type(src_path)
        if content_type is None:
            content_type = 'application/octet-stream'
        data = {
            "target_path": dest_path,
            "container_name": self.container_name,
        }
        max_execute_time,max_execute_retry = 300.0,10 # 最多5min,最多10次
        st,execute_time,retry_times = time.time(),0,0
        timeout,error_message = 180,""

        while execute_time < max_execute_time and retry_times < max_execute_retry:
            try:
                with open(src_path, 'rb') as local_file:
                    files = {'file': (os.path.basename(dest_path), local_file, content_type)}
                    response = self.session.post(f'{self.host}/upload',
                                            headers={"ROUTE-KEY": self.route_key},
                                            data=data,
                                            files=files,
                                            timeout=timeout)
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("status") == "Success":
                        return 
            except requests.exceptions.RequestException as e:
                error_message = repr(e)
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
        print(
            f"[ERROR][COPY TO CONTAINER ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {self.route_key}, data: {data}, files: {files}, error_message: {error_message}"
        )

    def create_file(self, file_path: str, content: str):
        # create a local file with the content
        data = {
                "content": content,
                "container_name": self.container_name,
                "path": f"{file_path}"
        }
        max_execute_time,max_execute_retry = 300.0,10 # 最多5min,最多10次
        st,execute_time,retry_times = time.time(),0,0
        timeout,error_message = 180,""
        while retry_times < max_execute_retry and execute_time < max_execute_time:
            try:
                response = self.session.post(f'{self.host}/write_file',
                                        headers={"ROUTE-KEY": self.route_key},
                                        json=data,
                                        timeout = timeout)
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("status") == "Success": return 
            except requests.exceptions.RequestException as e:
                error_message = repr(e)
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(1)
        print(
            f"[ERROR][CREATE FILE ERROR](retry_times:{retry_times})(execute_time: {execute_time})"
            f"route_key: {self.route_key}, input: {[data]}, timeout: {timeout}, error: {error_message}"
        )

    def setup_env(self):
        try:
            if self.swebench_verified:
                self.setup_env_swebench()
            else:
                self.setup_env_trainset()
            return "SUCCESS"
        except Exception as e:
            print(f"[ERROR]Error setting up environment: {repr(e)}")
            return "ERROR"

    def setup_env_trainset(self):
        # create a symlink from repo_path/.venv to /root/.venv
        self.run(f"ln -s {self.repo_path}/.venv {self.alt_path}/.venv")
        self.run(f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python")
        self.run(f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python3")
        self.run(f"find {self.repo_path}/.venv/bin -type f -executable -exec ln -sf {{}} {self.alt_path}/.local/bin/ \\;")

        # install required packages
        self.run("uv pip install chardet -i https://mirrors.aliyun.com/pypi/simple/")

        # clean cache file. also delete pycache and pyc.
        self.run("find . -name '*.pyc' -delete")
        self.run("find . -name '__pycache__' -exec rm -rf {} +")
        self.run("find /r2e_tests -name '*.pyc' -delete")
        self.run("find /r2e_tests -name '__pycache__' -exec rm -rf {} +")

        # move all skip files (if present) to /root
        for skip_file in SKIP_FILES_NEW:
            self.run(f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}")


        # r2e_tests are in the / directory, move them to /root
        self.run(f"mv /r2e_tests {self.alt_path}/r2e_tests")
        # make a softlink for /root/r2e_tests (if present)
        self.run(f"ln -s {self.alt_path}/r2e_tests {self.repo_path}/r2e_tests")

    def setup_env_swebench(self):
        try:
            # make the run_tests.sh executable
            self.run("chmod +x /run_tests.sh")
            
            # the run_test is in the "/" directory for swebench dockers
            self.alt_path = ("/"  )

            # make symlink of conda env to /root/.venv
            self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")

            # install chardet
            self.run("python -m pip install chardet -i https://mirrors.aliyun.com/pypi/simple/")
            return "SUCCESS"
        except Exception as e:
            print(
                f"[ERROR]Error setting up swebench environment: {repr(e)} @ {self.docker_image}"
            )
            return "ERROR"


    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir=None
    ) -> tuple[str, str]:
        """
        General method to execute code or commands in the container, with a timeout.

        :param code: The code or command to execute.
        :param args: Arguments to pass to the code/script.
        :param workdir: The working directory inside the container (optional).
        :return: A tuple containing (output, error_message). If no error, error_message is the exit code (int).
        """
        command = f"timeout {timeout} {code} {args}"

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Notice we do NOT set tty=True here
                future = executor.submit(
                    self.execute,
                    command=["/bin/sh", "-c", command],
                    workdir=self.repo_path if workdir is None else workdir,
                    environment={"PATH": DOCKER_PATH},
                    timeout=timeout
                )
                exec_result = future.result(timeout=timeout + 5)

            # Retrieve output and exit code
            if isinstance(exec_result, str): # 执行服务发生了问题
                print(f"[ERROR][服务执行有问题]command: {[command]}, exec_result({type(exec_result)}): {exec_result}")
                stdout = exec_result
                if exec_result.startswith('ERROR: '): error_code = -1
                else: error_code = 0
            elif isinstance(exec_result, dict): # 执行成功
                stdout = self.clean_stdout(exec_result["stdout"] + exec_result["stderr"])
                error_code = exec_result["exit_code"]
            else:
                raise ValueError(f'type(exec_result): {type(exec_result)}, exec_result: {exec_result}')

            if error_code == 124: # 运行超时
                print(f"[ERROR][运行超时]command: {[command]}, Internal Timeout: {timeout}s")
                return f"The command took too long to execute (>{timeout}s)", "-1"
            elif error_code != 0 and error_code != 1: # 0是成功, 1是通用错误，这个需要check一下。
                # output: [{'stdout': '', 'stderr': '/bin/sh: 1: cannot open /parameter: No such file\n', 'exit_code': 2}]  # TODO：这里可以设计不同的reward
                return stdout, str(error_code)
            else:
                # Remove ANSI escape codes and \r characters
                stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
                return stdout, str(error_code)

        ## timeout
        except concurrent.futures.TimeoutError:
            print(f"[ERROR]command: {[command]}, Timeout: {timeout}s")
            return f"The command took too long to execute (>{timeout}s)", "-1"

        except Exception as e:
            return f"Error: {repr(e)}", "-1"

    def clean_stdout(self,stdout):
        # 清理空的STDOUT和STDERR标签
        # 删除空的STDOUT标签（如果后面没有内容）
        stdout = re.sub(r"\[STDOUT\]\n\n \n\n\[STDERR\]", "[STDERR]", stdout)
        stdout = re.sub(r"\[STDOUT\]\n\n\[STDERR\]", "[STDERR]", stdout)
        stdout = re.sub(r"\[STDOUT\]\n\n \n", "", stdout)
        stdout = re.sub(r"\[STDOUT\]\n\n", "", stdout)
        # 清理多余的换行符
        stdout = re.sub(r"\n\n\[STDERR\]\n\n", "\n[STDERR]\n", stdout)
        stdout = re.sub(r"\n\n\[STDERR\]", "\n[STDERR]", stdout)
        # 如果STDERR也是空的，删除整个标签
        stdout = re.sub(r"\[STDERR\]\n\n$", "", stdout)
        stdout = re.sub(r"\[STDERR\]\n$", "", stdout)
        return stdout


    def demux_run(
        self, code: str, timeout: int = CMD_TIMEOUT, args: str = "", workdir=None
    ) -> tuple[str, str]:
        command = f"timeout {timeout} {code} {args}"
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Set demux=True to get separate stdout and stderr streams
                future = executor.submit(
                    self.container.exec_run,
                    cmd=command,
                    workdir=self.repo_path if workdir is None else workdir,
                    demux=True,  # This is the key change
                    environment={"PATH": DOCKER_PATH},
                )
                exec_result = future.result(timeout=timeout + 5)

            # Unpack the result - when demux=True, output is a tuple of (stdout_data, stderr_data)
            output_data, error_data = exec_result.output
            error_code = exec_result.exit_code

            # Handle None cases and decode the outputs
            stdout = (
                output_data.decode("utf-8", errors="replace") if output_data else ""
            )
            stderr = error_data.decode("utf-8", errors="replace") if error_data else ""

            if error_code != 0:
                print(
                    f"[ERROR] Error: Exit code {error_code} \nStdout Message: {stdout}, \nError Message: {stderr}"
                )
                return stdout, stderr, f"Error: Exit code {error_code}"

            return stdout, stderr, str(error_code)
        except Exception as e:
            return f"Error: {repr(e)}", f"Error: {repr(e)}", "-1"



    @DeprecationWarning  # TODO: remove dependency on this method with new dockers
    def read_file(self, rel_file_path: str) -> str:
        output, _ = self.run(f"cat /{self.alt_path}/{rel_file_path}")
        return output

    def run_tests(self) -> str:
        output, _= self.run(f"bash {self.alt_path}/run_tests.sh", timeout=300)
        # Remove ANSI escape codes and \r characters
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        return output

    def demux_run_tests(self) -> tuple[str, str, str]:
        stdout, stderr, error_code = self.demux_run(
            f"bash {self.alt_path}/run_tests.sh"
        )
        # Remove ANSI escape codes and \r characters
        stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
        stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", stderr)
        return stdout, stderr, error_code

    def checkout(self, commit_hash: str) -> str:
        output, _ = self.run(f"git checkout {commit_hash}")
        return output

    def get_patch(self) -> str:
        """
        Get the diff of the current state of the repository.
        """
        # git add -A && git diff --cached
        # self.run("git add -A")
        output, _ = self.run("git add -A && git diff --cached")
        # output, _ = self.run("git diff")
        return output


    def apply_patch(self, patch: str) -> str:
        # store the patch locally in a file identifiable by docker container id and timestamp
        # must contain unique patch name with both timestamp and docker image name
        uuid_ = uuid.uuid4()
        patch_path = f"{self.container_name}_{uuid_}.patch"
        patch_path = os.path.join("/tmp", patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        # copy the patch to / of the container
        self.copy_to_container(patch_path, f"/{patch_path}")
        # apply the patch
        output, _ = self.run(f"git apply --whitespace=fix /{patch_path}")
        return output

    def reverse_patch(self, patch: str) -> str:
        # store the patch locally in a file identifiable by docker container id and timestamp
        patch_path = f"{self.container_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.patch"
        patch_path = os.path.join("/tmp", patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        # copy the patch to / of the container
        self.copy_to_container(patch_path, f"/{patch_path}")
        # apply the patch
        output = self.run(f"git apply -R /{patch_path}")
        return output

    def get_logs_eval(
        self, test_spec: TestSpec, content: str
    ) -> tuple[dict[str, str], bool]:
        """
        Retrieve evaluation results for a task instance from its corresponding log file

        Args:
            log_fp (str): path to log file
        Returns:
            bool: whether the patch applied successfully
            dict: status map

        modified from swebench/harness/grading.py
        """
        repo = test_spec.repo
        version = test_spec.version
        log_parser = MAP_REPO_TO_PARSER[repo]
        test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        # print(f"[DEBUG]repo: {repo}")
        # print(f"[DEBUG]version: {version}")
        # print(f"[DEBUG]MAP_REPO_TO_PARSER keys: {list(MAP_REPO_TO_PARSER.keys())}")
        # print(f"[DEBUG]MAP_REPO_VERSION_TO_SPECS keys: {list(MAP_REPO_VERSION_TO_SPECS.keys())}")

        if isinstance(test_cmd, list):
            test_cmd = test_cmd[-1]
        # print(f"[DEBUG]test_cmd: {test_cmd}")
        # print(f"[DEBUG]content contains test_cmd: {test_cmd in content}")

        # with open(log_fp) as f:
        # # TODO fix constant here
        bad_codes = list(
            filter(
                lambda x: x in content,
                [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                ],
            )
        )
        if bad_codes:
            print(f"[ERROR]Bad code found in log: {bad_codes}")
            return {}, False

        # elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
        #     # Test patch did not apply (should not happen at all)
        #     print("Test patch did not apply")
        #     return {}, False

        # Get status map of evaluation results
        # content = content.split(test_cmd)[-1]

        self.logger.info(f"using swebench log_parser for repo: {repo}")
        try:
            return log_parser(content, test_spec), True
        except Exception as e:
            print(f"[ERROR]Error parsing logs for repo {repo}: {repr(e)}")
            print(f"[ERROR]Content preview: {content[:500]}...")
            return {}, False

    def parse_logs(self, log_output: str) -> dict:
        if self.swebench_verified:
            parsed_output, patch_apply_success = self.get_logs_eval(
                self.test_spec, log_output
            )
            return parsed_output
        else:
            return parse_log_fn(f"{self.repo_name}")(log_output)

    def _calculate_reward_swebench(self, get_test_output=False) -> float:
        # gt_test_patch = self.commit.get_patch(test_file=True,non_test_file=False)
        # self.apply_patch(gt_test_patch)

        # 原始run_tests.sh内容
        out, _ = self.run("cat /run_tests.sh", 300)  # run the tests after applying the patch

        # 注释掉conda相关命令的run_tests.sh内容
        self._rm_conda_in_swebench()
        out, _ = self.run("cat /run_tests.sh", 300)  # run the tests after applying the patch

        # 运行run_tests.sh
        out, _ = self.run("pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && /run_tests.sh", timeout=1800)  # run the tests after applying the patch

        # get eval logs
        eval_status_map, found = self.get_logs_eval(self.test_spec, out)

        eval_ref = {
            KEY_INSTANCE_ID: self.test_spec.instance_id,
            FAIL_TO_PASS: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: self.test_spec.PASS_TO_PASS,
        }
        report = get_eval_tests_report(
            eval_status_map, eval_ref, eval_type=get_eval_type(self.test_spec)
        )
        success = get_resolution_status(report) == ResolvedStatus.FULL.value
        if get_test_output:
            return success, out
        return int(success)
    
    def _rm_conda_in_swebench(self):
        """
        注释掉 /run_tests.sh 中与环境创建相关的命令
        包括 conda activate、source activate、pip install 等
        """
        try:
            # 读取当前的 /run_tests.sh 文件
            out_file, _ = self.run("cat /run_tests.sh", 300)
            if not out_file:
                print("[ERROR]无法读取 /run_tests.sh 文件")
                return False
            
            # 定义需要注释的环境相关命令
            env_commands_to_comment = [
                # conda 相关
                r'^conda activate',
                r'^source /opt/miniconda3/bin/activate',
                r'^source.*activate',
                
                # pip install 相关
                r'^python -m pip install',
                r'^pip install',
                r'^uv pip install',
                r'^poetry install',
                r'^pipenv install',
                
                # setup.py install 相关
                r'^python setup\.py install',
                r'^setup\.py install',
                r'^python.*setup\.py.*install',
            ]
            
            lines = out_file.split('\n')
            modified_lines = []
            commented_count = 0
            
            for line in lines:
                should_comment = False
                
                # 检查是否需要注释
                for pattern in env_commands_to_comment:
                    if re.search(pattern, line.strip()):
                        should_comment = True
                        break
                
                if should_comment and line.strip() and not line.strip().startswith('#'):
                    # 添加注释符号
                    modified_lines.append(f"# {line}")
                    commented_count += 1
                    # self.logger.info(f"注释环境命令: {line.strip()}")
                else:
                    modified_lines.append(line)
            
            # 写回文件
            modified_content = '\n'.join(modified_lines)
            
            # 创建临时文件
            temp_content = f"""#!/bin/bash
{modified_content}"""
            
            # 写入临时文件然后复制
            self.create_file("/tmp/run_tests_modified.sh", temp_content)
            self.run("cp /tmp/run_tests_modified.sh /run_tests.sh", 300)
            self.run("chmod +x /run_tests.sh", 300)
            
            # self.logger.info(f"成功注释了 {commented_count} 个环境相关命令")
            return True
            
        except Exception as e:
            print(f"[ERROR]注释环境命令时出错: {repr(e)}")
            return False
    
    def _backup_and_restore_run_tests(self, action='backup'):
        """
        备份或恢复 /run_tests.sh 文件
        
        Args:
            action: 'backup' 或 'restore'
        """
        try:
            if action == 'backup':
                # 备份原始文件
                self.run("cp /run_tests.sh /run_tests.sh.backup", 300)
                self.logger.info("已备份原始 /run_tests.sh 文件")
                return True
            elif action == 'restore':
                # 恢复原始文件
                self.run("cp /run_tests.sh.backup /run_tests.sh", 300)
                self.run("chmod +x /run_tests.sh", 300)
                self.logger.info("已恢复原始 /run_tests.sh 文件")
                return True
            else:
                print(f"[ERROR]未知操作: {action}")
                return False
        except Exception as e:
            print(f"[ERROR]备份/恢复文件时出错: {repr(e)}")
            return False
    def _print_unit_test_bash(self):
        text = ''
        # 打印当前目录结构 - 使用find和ls替代tree命令
        try:
            # 首先尝试使用tree命令（如果可用）
            out_tree, error_code = self.run("tree -I 'node_modules|__pycache__|*.pyc|.git|.pytest_cache|.mypy_cache|.coverage|htmlcov|dist|build|*.egg-info|.venv|venv|env|.env' -a", 300)
            if error_code != "0":
                # 如果tree命令不可用，使用find和ls的组合
                out_ls, _ = self.run("ls -la", 300)
                out_find, _ = self.run("find . -type f -name '*.py' -o -name '*.sh' -o -name '*.txt' -o -name '*.json' | head -20", 300)
                out_tree = f"ls -la output:\n{out_ls}\n\nfind . -type f output (first 20):\n{out_find}"
        except Exception as e:
            # 如果出现异常，使用基本的ls命令
            out_tree, _ = self.run("ls -la", 300)
            out_tree = f"ls -la output (fallback):\n{out_tree}"
        text += f'当前目录的树状结构: [TREE_START]\n{out_tree}\n[TREE_END]\n'
        
        # 打印swe-bench中的单测sh文件
        if self.swebench_verified:
            out_file, _ = self.run("cat /run_tests.sh", 300)  # run the tests after applying the patch
            print(f'\n[DEBUG]/run_tests.sh内容: [CONTENT_START]\n{out_file}\n[CONTENT_END]')
        # 打印r2e中的单测sh文件
        else:
            out_file, _ = self.run(f"cat {self.alt_path}/run_tests.sh", 300)  # run the tests after applying the patch
            print(f'\n[DEBUG]{self.alt_path}/run_tests.sh中的内容: [CONTENT_START]\n{out_file}\n[CONTENT_END]')
        text += f'\n\n[DEBUG]/run_tests.sh中的内容: [CONTENT_START]\n{out_file}\n[CONTENT_END]'

        # 提取各种测试命令并显示具体的测试文件内容
        import re
        
        # 支持多种测试命令的正则表达式
        test_patterns = [
            r'pytest\s+[^\n]*\.py',  # pytest 命令
            r'bin/test\s+[^\n]*\.py',  # 项目自定义测试脚本
            r'python\s+-m\s+pytest\s+[^\n]*\.py',  # python -m pytest
            r'python\s+-m\s+test\s+[^\n]*\.py',  # python -m test
            r'nosetests\s+[^\n]*\.py',  # nosetests
            r'unittest\s+[^\n]*\.py',  # unittest
            r'\./tests/runtests\.py\s+[^\n]*',  # Django runtests.py
            r'manage\.py\s+test\s+[^\n]*',  # Django manage.py test
            r'django-admin\s+test\s+[^\n]*',  # django-admin test
            r'tox\s+[^\n]*\.py',  # tox 命令
            r'make\s+test\s+[^\n]*',  # make test
            r'npm\s+test\s+[^\n]*',  # npm test
            r'yarn\s+test\s+[^\n]*',  # yarn test
        ]
        
        all_test_matches = []
        for pattern in test_patterns:
            matches = re.findall(pattern, out_file)
            all_test_matches.extend(matches)
        
        if all_test_matches:
            text += f'\n\n[DEBUG]找到的测试命令: [PATH_START]{all_test_matches}\n[PATH_END]'
            for i, test_cmd in enumerate(all_test_matches):
                # 提取测试文件路径 - 支持多种格式
                test_file_path = None
                
                # 尝试不同的提取模式
                patterns_to_try = [
                    r'pytest\s+[^\s]*\s+([^\s]+\.py)',  # pytest -rA file.py
                    r'bin/test\s+[^\s]*\s+([^\s]+\.py)',  # bin/test -C file.py
                    r'python\s+-m\s+pytest\s+[^\s]*\s+([^\s]+\.py)',  # python -m pytest file.py
                    r'python\s+-m\s+test\s+[^\s]*\s+([^\s]+\.py)',  # python -m test file.py
                    r'nosetests\s+[^\s]*\s+([^\s]+\.py)',  # nosetests file.py
                    r'unittest\s+[^\s]*\s+([^\s]+\.py)',  # unittest file.py
                    r'\./tests/runtests\.py\s+[^\s]*\s+([^\s]+)',  # ./tests/runtests.py --settings=test_sqlite model_forms
                    r'manage\.py\s+test\s+[^\s]*\s+([^\s]+)',  # manage.py test --settings=test_sqlite model_forms
                    r'django-admin\s+test\s+[^\s]*\s+([^\s]+)',  # django-admin test model_forms
                    r'tox\s+[^\s]*\s+--\s+([^\s]+\.py)',  # tox --current-env -epy39 -v -- file.py
                    r'make\s+test\s+[^\s]*\s+([^\s]+)',  # make test file
                    r'npm\s+test\s+[^\s]*\s+([^\s]+)',  # npm test file
                    r'yarn\s+test\s+[^\s]*\s+([^\s]+)',  # yarn test file
                ]
                
                for pattern in patterns_to_try:
                    test_file_match = re.search(pattern, test_cmd)
                    if test_file_match:
                        test_file_path = test_file_match.group(1)
                        break
                
                if test_file_path:
                    # 尝试读取测试文件内容
                    try:
                        test_file_content, _ = self.run(f"cat {test_file_path}", 300)
                        text += f'\n\n[DEBUG]测试文件内容 ({test_file_path}): [TEST_FILE_START]\n{test_file_content}\n[TEST_FILE_END]'
                    except Exception as e:
                        error_msg = f"无法读取测试文件 {test_file_path}: {str(e)}"
                        print(f'\n[DEBUG]error: {error_msg}')
                        text += f'\n\n[DEBUG]error: {error_msg}'
                else:
                    # 如果无法提取文件路径，至少显示命令
                    text += f'\n\n[DEBUG]测试命令 (无法提取文件路径): {test_cmd}'
        else:
            no_test_msg = "未找到测试命令"
            print(f'\n[DEBUG]error: {no_test_msg}')
            text += f'\n\n[DEBUG]error: {no_test_msg}'
        return text


    def _calculate_reward_r2e(self, get_test_output=False) -> float:
        # calculate reward based for r2e-edit dockers
        output = self.run_tests()
        # print(f"[DEBUG]最初始的output: \n{output}")
        # print(output)
        parse = self.parse_logs(output)
        parse = decolor_dict_keys(parse)
        try:
            expected_json = self.ds["expected_output_json"]
        except Exception as e:
            expected_json = self.read_file("expected_test_output.json")

        expected: dict = json.loads(expected_json)
        expected = decolor_dict_keys(expected)
        # 过滤掉空键，避免 KeyError
        parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys()) if k.strip()}
        expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys()) if k.strip()}

        # Compare
        if not parse or not expected:
            print(f'[ATTENTION][need check]parse or expected is null. parse: {parse}, expected: {expected}, output: {output}')
            return 0.0
        if len(parse) != len(expected):
            reward = 0.0
        else:
            # If ANY mismatch, reward = 0.0, else = 1.0
            match = True
            for k in parse.keys():
                if k not in expected:
                    print(f'[ATTENTION][need check]k not in expected. k: {k}, parse: {parse}, expected: {expected}, output: {output}')
                    continue
                if parse[k] != expected[k]:
                    match = False
                    break
            reward = 1.0 if match else 0.0
        # If the caller wants the test output as well, return (reward, output)
        if get_test_output:
            return reward, output
        return reward

    def _calculate_reward(self, get_test_output=False) -> float:
        if self.swebench_verified:
            # print('计算swebench的reward start：')
            reward = self._calculate_reward_swebench(get_test_output=get_test_output)
            print(f'计算swebench的reward end：{reward}')
        else:
            reward = self._calculate_reward_r2e(get_test_output=get_test_output)
            print(f'计算r2e的reward end：{reward}')
        return reward

    def reset(self):
        self.stop_container()
        self.start_container()

    def close(self):
        self.stop_container()

    def run_swebv_regression(
        self, run_tests_regression: str | None = None, timeout: int = 300
    ) -> str:
        # run the regression tests for swebench verified dockers
        # copy the 'run_tests_regression' thing from ds into the container at /run_tests_regression.sh
        if run_tests_regression is None:
            run_tests_regression = self.ds["run_tests_regression"]

        self.create_file("/run_tests_regression.sh", run_tests_regression)

        # make the script executable
        self.run("chmod +x /run_tests_regression.sh")

        # run the regression tests
        output, _ = self.run("/run_tests_regression.sh", timeout=timeout)
        return output
        # return swebench_parse(self.ds, output)

    def start_new_branch(self, branch_name: str = "exp") -> str:
        # ## save current branch-name
        # output, error_code = self.run("git branch --show-current")
        # self.current_branch = output.strip()
        # # new branch
        # output, error_code = self.run(f"git checkout -b {branch_name}")
        # # save commit hash

        output, _ = self.run(
            "git config --global user.email 'you@example.com'"
        )
        output, _ = self.run("git config --global user.name 'Your Name'")
        output, _ = self.run("git rev-parse HEAD")
        self.current_commit = output.strip()
        return output

    def commit_after_step(self, step_idx: int) -> str:
        # commit
        output, _ = self.run("git add .")
        output, _ = self.run(f"git commit -m '{step_idx}'")
        return output

    def undo_last_commit(self) -> str:
        # undo last commit
        output, _ = self.run("git reset --hard HEAD~1")
        return output

    def get_current_commit_hash(self) -> str:
        output, _ = self.run("git rev-parse HEAD")
        return output.strip()

    def soft_git_reset(self) -> str:
        # soft reset to saved commit
        output, _ = self.run(f"git reset --soft {self.current_commit}")

        # # checkout to saved branch
        # output, error_code = self.run(f"git checkout {self.current_branch}")

        return output
