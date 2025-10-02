from typing import Coroutine, Any, Tuple, Dict, Optional
import asyncio
import json
import httpx
from roll.agentic.env.base import BaseEnv
from roll.agentic.env.mcp.mcp_client import MCPClient
from roll.agentic.env.parse_action_utils import default_parser_action_func
from roll.agentic.env.mcp.sokoban_mcp_config import SokobanMCPEnvConfig
from mcore_adapter.utils import get_logger

logger = get_logger(__name__)

class SokobanMCPEnv(BaseEnv):
    def __init__(self, config: SokobanMCPEnvConfig, client: Optional[MCPClient] = None):
        super().__init__(config)
        self.client: Optional[MCPClient] = None
        self._connected: bool = False
        
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
        if client:
            self.client = client
        else:
            self.client = MCPClient(self.config.server_url) 
        
        self._last_obs = None
        self._last_info = {}    

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Any, dict]:
        """
        Resets the environment by calling the 'reset' tool via the MCP server.
        """
        logger.info(f"Resetting Sokoban environment with seed={seed}...")
        tool_name = "reset"
        tool_params = {"seed": seed} if seed is not None else {}
        
        try:
            # Use the helpers from the base class to run the async logic.
            # The base class will handle the call and the parsing pipeline.
            obs, _, _, info = self._run_async_logic(
                self._execute_and_parse(tool_name, tool_params)
            )
        except (httpx.ReadTimeout, httpx.ConnectError, ConnectionError, ValueError, json.JSONDecodeError) as e:
            error_message = f"Failed to reset the environment due to a server or network issue: {e}"
            logger.error(error_message)
            raise RuntimeError(error_message) from e
        except Exception as e:
            logger.error(f"FATAL: An unexpected critical error occurred during reset: {e}")
            raise RuntimeError("Environment failed to reset due to an unexpected error.") from e  
        
        self._last_obs = obs
        self._last_info = info
        return obs, info
        
    def parse_action(self, text: str) -> Optional[int]:
        """
        Parses a simple action string like "<answer>Up</answer>" and returns the action ID.
        Returns None if parsing fails.
        """
        return default_parser_action_func(text, self.config.action_pattern, self.config.action_lookup, self.config.special_token_list)

    def step(self, action: str) -> Tuple[Any, float, bool, bool, Dict]:
        """
        This step method mirrors the simple SokobanEnv's logic, but uses tool calls.
        """
        # 1. Parse the action string to get an action ID.
        action_info = self.parse_action(action)
        action_id = action_info.get("action")

        # 2. Handle parsing failure (e.g., LLM output is wrong)
        if action_id is None:
            # This logic is taken directly from the simple SokobanEnv
            metrics = {
                "action_is_effective": False,
                "action_is_valid": False, # This is the key signal for format_penalty
                "success": False, # Assuming no success on a failed parse
            }
            info = {
                "metrics": metrics,
                "raw_action_text": action,
            }
            # We use _last_obs to ensure the agent sees the same state again
            # We return a neutral reward of -0.1 to keep the same behavior as gym
            return self._last_obs, -0.1, False, False, info

        # 3. Handle parsing success: execute the action via a tool call
        tool_name = "play"
        tool_params = {"action": action_id}

        try:
            # This is the MCP equivalent of `GymSokobanEnv.step()`
            obs, terminated, truncated, info = self._run_async_logic(
                self._execute_and_parse(tool_name, tool_params)
            )
        except Exception as e:
            # Keep our robust error handling for network/server issues
            logger.error(f"Server/Network Error on action '{action}': {e}", exc_info=True)
            info = {
                "metrics": {
                    "action_is_effective": False,
                    "action_is_valid": False, # This is the key signal for EnvManager
                    "success": False,
                },
                "error_details": str(e),
            }
            # set truncated = true
            return "SYSTEM_ERROR", 0.0, False, True, info
        
        reward_from_server = info.get('reward_from_server', 0.0) 
        
        info.update({
            "tool_name": tool_name,
            "tool_params": tool_params,
            "raw_action_text": action,
        })
        
        # Update internal state and add action info for logging
        self._last_obs = obs
        self._last_info = info

        return obs, reward_from_server, terminated, truncated, info
        
    def parse_action(self, text) -> Dict[str, Any]:
        return default_parser_action_func(text, self.config.action_pattern, self.config.action_lookup, self.config.special_token_list)

    def get_all_actions(self) -> list[str]:
        return list(self.config.action_lookup.values())    
    
    def render(self, mode="text") -> Any:
        if mode == "text":
            return self._last_obs or "The environment has not been reset yet."
        else:
            raise NotImplementedError(f"Render mode {mode} is not implemented")
    
    def close(self):
        """Closes the connection to the MCP server."""
        if self._connected:
            self._run_async_logic(self._disconnect())
        # Ensure the event loop is closed if this class created it.
        if not asyncio.get_event_loop().is_running():
            self._event_loop.close()
    
    async def _connect(self):
        if not self._connected:
            if self.client is None:
                raise RuntimeError("Client has not been initialized.")       
            await self.client.__aenter__()
            self._connected = True
    
    async def _disconnect(self):
        if self.client and self._connected:
            await self.client.__aexit__(None, None, None)
            self._connected = False   
            
    def _run_async_logic(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Runs an async coroutine in the managed event loop."""
        if self._event_loop.is_running():
            # This case handles environments where the outer framework is already async.
            future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
            return future.result()
        else:
            # This case handles a purely synchronous script.
            return self._event_loop.run_until_complete(coro)
        
    async def _execute_and_parse(self, tool_name: str, tool_params: Dict) -> Tuple[Any, bool, Dict]:
        """Async helper to connect, call a tool, and parse its response."""
        if not self._connected:
            await self._connect()
        # It's the subclass's job to parse the raw response.
        raw_res = await self.client.call_tool(tool_name, tool_params)
        return self._parse_tool_response(raw_res)
    
    def _parse_tool_response(self, response: Any) -> Tuple[Any, bool, bool, Dict]:
        """
        Default tool response parser. It extracts the text content from the
        tool's raw response, assumes it's a JSON string, and then calls
        a new abstract method `process_parsed_json` to create the final output.
        This separates the "unwrapping" logic from the "interpreting" logic.
        """
        text_content = next((item.text for item in getattr(response, "content", []) if getattr(item, "type", None) == "text"), None)
        if text_content is None:
            raise ValueError("Tool server response is empty or does not contain 'text' content.")
        
        logger.debug(f"Parsing tool response text: {text_content}")
        
        try:
            parsed_json = json.loads(text_content)
            return self._process_parsed_json(parsed_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse server response. Content was: '{text_content}'") from e
        
    def _process_parsed_json(self, data: Dict) -> Tuple[str, bool, bool, Dict]:
        """
        Processes the JSON data from the Sokoban tool (`reset` or `play`)
        into a text-based observation, a done flag, and an info dict.
        This implementation is based on the keys found in the test file.
        """  
        # Extract raw signals from the server's JSON response
        observation_text = data.get("Observation", "Error: No observation found.")
        server_reward = data.get("Reward", 0.0) # Default to 0.0 if not provided
        game_end = data.get("Game End", False)
        server_info = data.get("info", {})
        
        # Translate raw signals into standard terminated/truncated flags 
        game_success = server_info.get("success", False)    
        is_terminated = game_end and game_success
        is_truncated = game_end and not game_success
        
        info = {
            "metrics": server_info,
            "reward_from_server": server_reward
        }
        
        # Enhance the observation with a legend for the LLM
        full_observation = (
            f"{observation_text}\n\n"
            "Legend: P=Player, X=Box, O=Target, √=Box on Target, #=Wall, _=Empty"
        )
        return full_observation, is_terminated, is_truncated, info