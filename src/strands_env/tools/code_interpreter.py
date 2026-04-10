# Copyright 2025-2026 Horizon RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code sandbox toolkit using AWS Bedrock AgentCore Code Interpreter."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any

from aiolimiter import AsyncLimiter
from strands import tool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from strands_env.utils.aws import BotoClient


class CodeInterpreterQuotas:
    """Shared AWS quotas for Code Interpreter API operations.

    Notes:
        - Create one instance and pass it to all `CodeInterpreterToolkit` instances
        to enforce account-wide limits across concurrent sessions.
        - Manages three concerns:
            - Session semaphore: caps concurrent sessions (`session_concurrency`).
            - Rate limiters: caps API request initiation rate for start/invoke/stop
              (AWS TPS quotas) to prevent throttling errors.
            - Thread pool executor: sized to match `session_concurrency` so each session can
              have one in-flight blocking boto3 call without starving others.

    References:
        - [AWS Bedrock AgentCore default quotas](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html)
    """

    DEFAULT_SESSION_CONCURRENCY = 3000
    DEFAULT_START_TPS = 30
    DEFAULT_INVOKE_TPS = 300
    DEFAULT_STOP_TPS = 30

    def __init__(
        self,
        session_concurrency: int = DEFAULT_SESSION_CONCURRENCY,
        start_tps: float = DEFAULT_START_TPS,
        invoke_tps: float = DEFAULT_INVOKE_TPS,
        stop_tps: float = DEFAULT_STOP_TPS,
    ):
        """Initialize a `CodeInterpreterQuotas` instance."""
        self.session_semaphore = asyncio.Semaphore(session_concurrency)
        self.start_limiter = AsyncLimiter(start_tps, time_period=1)
        self.invoke_limiter = AsyncLimiter(invoke_tps, time_period=1)
        self.stop_limiter = AsyncLimiter(stop_tps, time_period=1)
        self.executor = ThreadPoolExecutor(max_workers=session_concurrency)

    def to_thread(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking function in the quotas thread pool (or default pool if no quotas)."""
        return asyncio.get_running_loop().run_in_executor(self.executor, partial(func, *args, **kwargs))

    async def _acquire_semaphore_with_warning(self, name: str) -> None:
        """Acquire session semaphore and warn if it blocked."""
        start = time.time()
        await self.session_semaphore.acquire()
        elapsed = time.time() - start
        if elapsed > 0.001:  # If acquisition took > 1ms, we likely waited
            logger.warning(
                f"{name} semaphore hit limit (max: {self.session_semaphore._value + 1}), "
                f"waited {elapsed:.3f}s"
            )

    async def _acquire_limiter_with_warning(self, limiter: AsyncLimiter, name: str) -> None:
        """Acquire rate limiter and warn if it blocked."""
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        if elapsed > 0.001:  # If acquisition took > 1ms, we likely waited
            logger.warning(f"{name} rate limiter hit limit, waited {elapsed:.3f}s")


class CodeInterpreterToolkit:
    """Code toolkit using AWS Bedrock AgentCore Code Interpreter.

    Notes:
        - Provides `execute_code` and `execute_command` tools for running Python code
          and shell commands in a sandboxed environment.
        - Uses a single shared agentcore session through session ID. Call
          `cleanup` when done to close the session.
    """

    CODE_INTERPRETER_ID = "aws.codeinterpreter.v1"

    def __init__(
        self,
        client: BotoClient,
        session_name: str = "strands-env",
        quotas: CodeInterpreterQuotas | None = None,
    ):
        """Initialize a `CodeInterpreterToolkit` instance.

        Args:
            client: boto3 client for bedrock-agentcore service.
            session_name: Name for the code interpreter session.
            quotas: Shared quotas for rate limiting, session concurrency, and thread pool.
                Create one `CodeInterpreterQuotas` instance and pass it to all toolkit
                instances to enforce account-wide limits.
        """
        self.session_name = session_name
        self.client = client
        self.session_id: str | None = None
        self.quotas = quotas or CodeInterpreterQuotas()
        self._session_lock = asyncio.Lock()

    async def start_session(self) -> None:
        """Start a code interpreter session if not already started (async, thread-safe)."""
        if self.session_id is None:
            async with self._session_lock:
                # Double-check after acquiring lock as another coroutine may have set it
                if self.session_id is not None:
                    return  # type: ignore[unreachable]

                await self.quotas._acquire_semaphore_with_warning("Session")
                await self.quotas._acquire_limiter_with_warning(self.quotas.start_limiter, "StartSession")
                try:
                    response = await self.quotas.to_thread(
                        self.client.start_code_interpreter_session,
                        codeInterpreterIdentifier=self.CODE_INTERPRETER_ID,
                        name=self.session_name,
                        sessionTimeoutSeconds=3600,
                    )
                except Exception:
                    self.quotas.session_semaphore.release()
                    raise
                self.session_id = response["sessionId"]

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by the specified number of spaces.

        Args:
            code: Code to indent.
            spaces: Number of spaces to add to each line.

        Returns:
            Indented code.
        """
        indent = ' ' * spaces
        lines = code.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)

    def _wrap_code_with_stdin(self, code: str, stdin: str) -> str:
        """Wrap code with stdin input handling.

        Args:
            code: The Python code to wrap.
            stdin: Input to provide via stdin.

        Returns:
            Wrapped code that handles stdin.
        """
        # Escape stdin for safe embedding in code string
        escaped_stdin = stdin.replace('\\', '\\\\').replace("'''", "\\'\\'\\'")

        wrapped_code = f"""import sys
from io import StringIO
import builtins

# Store input lines
_input_lines = '''{escaped_stdin}'''.split('\\n')
_input_index = [0]  # Use list for mutable reference in closure

# Override input() function
_original_input = builtins.input
def _mock_input(prompt=''):
    if _input_index[0] >= len(_input_lines):
        raise EOFError('No more input available')
    line = _input_lines[_input_index[0]]
    _input_index[0] += 1
    return line

builtins.input = _mock_input

# Override sys.stdin as fallback
sys.stdin = StringIO('''{escaped_stdin}''')

try:
    # Execute user code
{self._indent_code(code, 4)}
finally:
    # Restore original input function
    builtins.input = _original_input
"""
        return wrapped_code

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke the code interpreter and return parsed response."""
        await self.start_session()
        await self.quotas._acquire_limiter_with_warning(self.quotas.invoke_limiter, "InvokeCodeInterpreter")
        response = await self.quotas.to_thread(
            self.client.invoke_code_interpreter,
            codeInterpreterIdentifier=self.CODE_INTERPRETER_ID,
            sessionId=self.session_id,
            name=name,
            arguments=arguments,
        )
        # Parse the `EventStream` response from `invoke_code_interpreter`.
        for event in response.get("stream", []):
            if "result" in event:
                content = event["result"].get("content", [])
                if isinstance(content, list):
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    return "\n".join(texts) if texts else str(content)
                return str(content)

            for key in (
                "accessDeniedException",
                "conflictException",
                "internalServerException",
                "resourceNotFoundException",
                "serviceQuotaExceededException",
                "throttlingException",
                "validationException",
            ):
                if key in event:
                    return f"{key}: {event[key].get('message', key)}"

        return "No result returned."

    @tool
    async def execute_code(self, code: str) -> str:
        """Execute Python code and return the result.

        Args:
            code: The Python code to execute.

        Returns:
            Execution output text or error message.
        """
        return await self.invoke("executeCode", {"code": code, "language": "python"})

    @tool
    async def execute_code_with_stdin(self, code: str, stdin: str) -> str:
        """Execute Python code with stdin input and return the result.

        Args:
            code: The Python code to execute.
            stdin: Input to provide via stdin (newline-separated for multiple input() calls).

        Returns:
            Execution output text or error message.
        """
        wrapped_code = self._wrap_code_with_stdin(code, stdin)
        return await self.invoke("executeCode", {"code": wrapped_code, "language": "python"})

    @tool
    async def execute_command(self, command: str) -> str:
        """Execute a shell command and return the result.

        Args:
            command: The shell command to execute.

        Returns:
            Execution output text or error message.
        """
        return await self.invoke("executeCommand", {"command": command})

    async def cleanup(self) -> None:
        """Clean up code interpreter session."""
        if self.session_id:
            await self.quotas._acquire_limiter_with_warning(self.quotas.stop_limiter, "StopSession")
            try:
                await self.quotas.to_thread(
                    self.client.stop_code_interpreter_session,
                    codeInterpreterIdentifier=self.CODE_INTERPRETER_ID,
                    sessionId=self.session_id,
                )
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                self.quotas.session_semaphore.release()
            self.session_id = None