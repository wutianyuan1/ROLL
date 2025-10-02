# repo_env.py
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import gym


from roll.agentic.env.swe_env.src.agenthub.action import Action
from roll.agentic.env.swe_env.src.agenthub.utils.log import get_logger
from roll.agentic.env.swe_env.src.agenthub.observation import Observation
from roll.agentic.env.swe_env.src.agenthub.runtime.docker import DockerRuntime
from roll.agentic.env.swe_env.src.agenthub.agent.commands import ParseCommandBash

cmd_parser = ParseCommandBash()


@dataclass(frozen=True)
class EnvArgs:
    """Configure data sources and setup instructions for the environment in which we solve the tasks."""

    ds: Dict
    repo_path: Optional[str] = None
    docker_image: Optional[str] = None


class RepoEnv(gym.Env):
    def __init__(self, logger=None):
        # Get the logger
        if logger is None:
            self.logger = get_logger("RepoEnv")  # Pass the module name for clarity
        else:
            self.logger = logger
        # self.runtime = DockerRuntime(
        #     ds=args.ds, command=["/bin/bash", "-l"], logger=self.logger, route_key=args.ds.get("route_key", None)
        # )
        self.done = False
        self.observation = None
        self.state = None
        self.cmd_parser = ParseCommandBash()
        self.runtime = None
        self.container_name = None
        self.route_key = None
        self.retry_times = None
        self.state = 'init'
        self.done = False

    def reset(self, args: EnvArgs) -> Dict[str, Any]:
        """
        Resets the environment and returns an initial observation.
        """
        self.logger.info(f"Resetting RepoEnv ...")
        self.runtime = DockerRuntime(
            ds=args.ds, command=["/bin/bash", "-l"], logger=self.logger, route_key=args.ds.get("route_key", None)
        )
        self.container_name = self.runtime.container_name
        self.route_key = self.runtime.route_key
        self.retry_times = self.runtime.retry_times
        self.observation = "Environment reset"
        self.state = 'reset'
        self.done = False
        return self.observation  # self.get_observation()

    def add_commands(self, cmd_files: list[str]):
        """
        Adds command files to the environment by parsing them,
        copying them to the Docker container, and making them executable or sourced.

        Args:
            cmd_files: List of paths to command files.
        """
        cmds = []
        for cmd_file in cmd_files:
            current_file_path = Path(__file__).resolve()
            cmd_file = cmd_file.replace("./", f"{current_file_path.parent.parent.parent.parent.parent}/")
            # Parse commands from file
            parsed_commands = self.cmd_parser.parse_command_file(cmd_file)
            cmds.extend(parsed_commands)

            # Determine the file extension
            _, ext = os.path.splitext(cmd_file)

            # Get the base name of the command file
            cmd_name = os.path.basename(cmd_file)

            if ext == ".py" or self._is_shebang_script(cmd_file):
                # Python script or shebang script: copy, strip .py extension if applicable
                if ext == ".py":
                    container_cmd_name = cmd_name[:-3]  # Remove .py extension
                else:
                    container_cmd_name = cmd_name
                container_path = f"/usr/local/bin/{container_cmd_name}"
                self.runtime.copy_to_container(cmd_file, container_path)
                self.runtime.run(f"chmod +x {container_path}")

            elif ext == ".sh":
                # Bash script ending with .sh: copy, chmod, and source it
                container_cmd_name = cmd_name
                container_path = f"/usr/local/bin/{container_cmd_name}"
                self.runtime.copy_to_container(cmd_file, container_path)
                # self.runtime.run(f"chmod +x {container_path}")
                # Source the script inside the container
                self.runtime.run(f"bash -c 'source {container_path}'")

            else:
                # Bash script without shebang: copy, chmod, and source it
                container_cmd_name = cmd_name
                container_path = f"/usr/local/bin/{container_cmd_name}"
                self.runtime.copy_to_container(cmd_file, container_path)
                self.runtime.run(f"chmod +x {container_path}")
                # Source the script inside the container
                self.runtime.run(f"bash -c 'source {container_path}'")

        # Store the parsed commands for reference
        self.commands = cmds
        self.logger.info(f"Added {len(cmds)} commands to the environment.")

    def _is_shebang_script(self, cmd_file: str) -> bool:
        """
        Checks if the given file starts with a shebang (#!).

        Args:
            cmd_file: Path to the command file.

        Returns:
            True if the file starts with a shebang, False otherwise.
        """
        with open(cmd_file, "r") as file:
            first_line = file.readline().strip()
        return first_line.startswith("#!")

    def run_action(self, action: Action, timeout: int):
        # check for empty or no function call / action
        if not action.function_name:
            allowed_cmds = [x.name for x in self.commands]
            error = f"Invalid Action: input action must be one of allowed actions \n Allowed actions: {allowed_cmds}\n."
            return error, error, 0
        start_time = time.time()
        bash_output = None
        try:
            # Check if action is in allowed actions/commands
            action_name = action.function_name
            allowed_cmds = [x.name for x in self.commands]
            assert (
                action_name in allowed_cmds
            ), f"Invalid Action: input action must be one of allowed actions \n Allowed actions: {allowed_cmds} \n Input action: {action_name}\t"

            # Run action and return
            bash_cmd = action.to_bashcmd()
            print('\nbash_cmd: ', bash_cmd)
            bash_output, error_code = self.runtime.run(bash_cmd, timeout=timeout)
            print('\nbash_output: ', bash_output)
            print('\nerror_code: ', error_code)
            return bash_output, error_code, time.time()-start_time
        except Exception as e:
            # Capture the error message as observation
            obs = str(e)
            error = f"action.function_name: {action.function_name}, Exception occurred: {obs}"
            self.logger.error(f"[run_action error]: {error}")
            return obs,-1, time.time()-start_time
        
    def step(
        self, action: Action, timeout: int
    ) -> Tuple[Observation, int, bool, Dict[str, Any]]:
        """
        Executes an action (command) in the Docker container.
        Runs an action proposed by the agent in the environment and returns the corresponding output.

        Args:
            action: command to run in bash shell

        Returns:
            observation:  output from container
            reward: Always set to 0
            done: whether task is over
            info: additional information (e.g. debugging information)
        """
        reward = 0
        bash_output, error_code, total_time = self.run_action(action, timeout=timeout)
        self.observation = Observation(bash_output, error_code, action)
        if "finish" in action.function_name.lower() or self.done:
            self.done = True
            # reward = self.calculate_reward(self.observation)
        info = {"total_time": total_time}
        # print(f'[RepoEnv.step]observation: {self.observation}')
        return self.observation, reward, self.done, info


    @property
    def _observation(self) -> Dict[str, Any]:
        return {"output": self.observation}

    @property
    def _state(self) -> Dict[str, Any]:
        return {"state": self.state}

    def setup_action_space(self):
        """add different allowed actions"""
        pass

    def add_actions(self, actions: list[dict]) -> None:
        """add different tools from the agent here"""
        pass

    def calculate_reward(self, obs: Observation) -> int:
        """
        Basic reward calculation based on command success.
        """
        self.runtime.calculate_reward(obs)
        return 0  # TODO

    def check_done(self) -> bool:
        return self.done  # Customize to set completion condition

    def close(self):
        self.runtime.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns the statistics of the environment.
        """
        return self.runtime.ds

