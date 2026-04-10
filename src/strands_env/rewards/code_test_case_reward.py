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

"""Reward function for algorithmic coding problems with test case validation.

Extracts Python code from agent response, executes it against hidden test cases,
and returns reward as the proportion of test cases passed.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult
from strands_env.tools import CodeInterpreterToolkit
from strands_env.utils.aws import get_client

if TYPE_CHECKING:
    from botocore.client import BaseClient

logger = logging.getLogger(__name__)


class CodeTestCaseReward(RewardFunction):
    """Reward based on proportion of hidden test cases passed.

    Args:
        client: AWS Bedrock AgentCore client for code execution. If None, creates default client.
        extract_last_code_block: If True, only use the last ```python block; if False, use first (default True).

    Test Case Format:
        The `ground_truth` in `action.task_context` should be a dict with:
        ```python
        {
            "inputs": ["test_input_1", "test_input_2", ...],
            "outputs": ["expected_output_1", "expected_output_2", ...],
        }
        ```
        Each input/output is a string representing the full stdin/stdout for that test case.

    Reward Calculation:
        - Base reward = (number of passed test cases) / (total test cases)
        - Returns 0.0 if code cannot be extracted or has syntax errors
        - Returns proportional reward in [0.0, 1.0] based on test pass rate

    Notes:
        - Extracts code from ```python...``` blocks in the agent's final response
        - Reuses a single code interpreter session across all test cases in the trajectory
        - Compares outputs with exact string matching (strips whitespace)
        - Handles execution errors gracefully (counts as test failure)
        - Relies on AWS server-side timeouts for code execution
        - Call `cleanup()` when done to close the code interpreter session
    """

    def __init__(
        self,
        client: BaseClient | None = None,
        extract_last_code_block: bool = True,
    ) -> None:
        """Initialize a `CodeTestCaseReward` instance."""
        self.client = client or get_client("bedrock-agentcore")
        self.extract_last_code_block = extract_last_code_block
        # Initialize toolkit once and reuse across all test cases in the trajectory
        self._toolkit = CodeInterpreterToolkit(client=self.client, session_name="code-reward-eval")

    def extract_code(self, text: str) -> str | None:
        """Extract Python code from markdown code blocks.

        Args:
            text: Text containing ```python code blocks.

        Returns:
            Extracted code string or None if no code block found.
        """
        # Pattern to match ```python ... ``` blocks
        pattern = r"```python\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        if not matches:
            # Try without language specifier
            pattern = r"```\s*\n(.*?)\n```"
            matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            return None

        # Return last or first match based on configuration
        return matches[-1] if self.extract_last_code_block else matches[0]

    async def execute_with_input(self, code: str, test_input: str, toolkit: CodeInterpreterToolkit) -> str:
        """Execute code with given input using code interpreter.

        Args:
            code: Python code to execute.
            test_input: Input to provide via stdin.
            toolkit: CodeInterpreterToolkit instance for execution.

        Returns:
            Execution output or error message.
        """
        try:
            # Use the execute_code_with_stdin tool (handles wrapping internally)
            result = await toolkit.execute_code_with_stdin(code, test_input)
            return result
        except Exception as e:
            return f"EXECUTION_ERROR: {type(e).__name__}: {str(e)}"

    def compare_outputs(self, expected: str, actual: str) -> bool:
        """Compare expected and actual outputs.

        Args:
            expected: Expected output string.
            actual: Actual output string from execution.

        Returns:
            True if outputs match (after stripping whitespace).
        """
        # Normalize: strip leading/trailing whitespace and normalize line endings
        expected_normalized = expected.strip().replace("\r\n", "\n")
        actual_normalized = actual.strip().replace("\r\n", "\n")

        return expected_normalized == actual_normalized

    async def run_test_cases(
        self,
        code: str,
        test_inputs: list[str],
        test_outputs: list[str],
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Run code against multiple test cases.

        Args:
            code: Python code to test.
            test_inputs: List of input strings.
            test_outputs: List of expected output strings.

        Returns:
            Tuple of (passed_count, total_count, test_results_details).
        """
        if len(test_inputs) != len(test_outputs):
            logger.error("Mismatched test inputs/outputs lengths: %d vs %d", len(test_inputs), len(test_outputs))
            return 0, len(test_inputs), []

        passed = 0
        results = []

        for idx, (test_input, expected_output) in enumerate(zip(test_inputs, test_outputs)):
            actual_output = await self.execute_with_input(code, test_input, self._toolkit)

            # Check if output matches
            is_passed = self.compare_outputs(expected_output, actual_output)
            if is_passed:
                passed += 1

            results.append(
                {
                    "test_case": idx,
                    "passed": is_passed,
                    "expected": expected_output[:200],  # Truncate for logging
                    "actual": actual_output[:200],
                }
            )

        return passed, len(test_inputs), results

    @override
    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Compute reward based on test case pass rate.

        Args:
            action: The action containing task context with ground truth test cases.
            step_result: The step result containing agent's response.

        Returns:
            RewardResult with reward in [0.0, 1.0] and detailed test results in info.
        """
        # Get final response
        content = step_result.observation.final_response
        if content is None:
            return RewardResult(reward=0.0, info={"reason": "no_final_response"})

        # Extract code from response
        code = self.extract_code(content)
        if code is None:
            return RewardResult(
                reward=0.0,
                info={
                    "reason": "no_code_block_found",
                    "response_sample": content[:500],
                },
            )

        # Get test cases from ground truth
        ground_truth = action.task_context.ground_truth
        if not isinstance(ground_truth, dict):
            return RewardResult(
                reward=0.0,
                info={
                    "reason": "invalid_ground_truth_format",
                    "ground_truth_type": type(ground_truth).__name__,
                },
            )

        test_inputs = ground_truth.get("inputs", [])
        test_outputs = ground_truth.get("outputs", [])

        if not test_inputs or not test_outputs:
            return RewardResult(
                reward=0.0,
                info={
                    "reason": "no_test_cases",
                    "inputs_count": len(test_inputs),
                    "outputs_count": len(test_outputs),
                },
            )

        # Run test cases
        try:
            passed, total, test_results = await self.run_test_cases(code, test_inputs, test_outputs)
        except Exception as e:
            logger.error("Failed to run test cases: %s: %s", type(e).__name__, str(e))
            return RewardResult(
                reward=0.0,
                info={
                    "reason": "test_execution_failed",
                    "error": f"{type(e).__name__}: {str(e)}",
                },
            )

        # Calculate reward
        reward = passed / total if total > 0 else 0.0

        return RewardResult(
            reward=reward,
            info={
                "passed": passed,
                "total": total,
                "pass_rate": reward,
                "code_extracted": code[:500],  # First 500 chars for debugging
                "test_results": test_results[:5],  # First 5 test results for debugging
            },
        )

    async def cleanup(self) -> None:
        """Clean up the code interpreter session.

        Should be called when done with the reward function to release resources.
        """
        await self._toolkit.cleanup()
