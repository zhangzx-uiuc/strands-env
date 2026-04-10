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

"""
Agentic generation function for algorithmic Python coding RL training.

Uses CodeSandboxEnv with tool interaction (`execute_code`) for iterative
problem-solving and code execution.
"""

import logging

from slime.rollout.sglang_rollout import GenerateState  # type: ignore
from slime.utils.types import Sample  # type: ignore
from strands_sglang import get_client_from_slime_args

from strands_env.core.models import sglang_model_factory
from strands_env.core.types import Action, TaskContext
from strands_env.environments.code_sandbox import CodeSandboxEnv
from strands_env.rewards.code_test_case_reward import CodeTestCaseReward
from strands_env.utils.aws import get_client
from strands_env.utils.slime import RolloutLogger

# export for slime's --custom-rollout-log-function-path
log_rollout_metrics = RolloutLogger(n_rollouts_per_step=3, log_per_tool_metrics=True).log_rollouts

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert Python programmer solving algorithmic coding problems.

Guidelines:
- You have access to the `execute_code` tool to test and debug your solution
- You MUST use the tool to test given input test cases before submitting your final solution
- Your final solution MUST be provided in a ```python code block
- Your code should read input from stdin and write output to stdout
- Make sure your solution handles all edge cases correctly

Example format:
```python
# Your solution here
```
""".strip()

MAX_TOOL_ITERS = 10
MAX_TOOL_CALLS = None

ROLE_ARN = "arn:aws:iam::022992367762:role/CrossAccountAccessRole-prod"

async def generate_and_rm(args, sample: Sample, sampling_params) -> Sample:
    """Generate and compute rewards using CodeSandboxEnv with test case validation.

    This is an agentic approach where the model can iteratively use the execute_code
    tool to test and refine its solution.

    Args:
        args: Arguments containing model checkpoint and server configuration.
        sample: Sample with prompt and label (test cases).
        sampling_params: Sampling parameters for generation.

    Returns:
        Sample with response, tokens, loss mask, reward, and metrics.
    """
    assert not args.partial_rollout, "Partial rollout not supported."

    state = GenerateState(args)

    # Parse test cases from label (ALL are hidden for reward computation)
    label = sample.label
    if not isinstance(label, dict):
        logger.error("Invalid label format: expected dict, got %s", type(label).__name__)
        sample.reward = 0.0
        sample.status = Sample.Status.COMPLETED
        return sample

    hidden_inputs = label.get("inputs", [])
    hidden_outputs = label.get("outputs", [])

    # Check we have test cases for reward computation
    if not hidden_inputs or not hidden_outputs:
        logger.error("No test cases found for sample")
        sample.reward = 0.0
        sample.status = Sample.Status.COMPLETED
        return sample

    # Use prompt as-is (visible examples already included)
    prompt = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[0]["content"]

    # Create model factory and clients
    model_factory = sglang_model_factory(
        tokenizer=state.tokenizer,
        client=get_client_from_slime_args(args, timeout=900.0),
        sampling_params=sampling_params,
    )
    config = Config(
        max_pool_connections=1024,
    )
    bedrock_client = get_client("bedrock-agentcore", role_arn=ROLE_ARN, config=config)

    # Create reward function with hidden test cases
    reward_fn = CodeTestCaseReward(client=bedrock_client)

    env = CodeSandboxEnv(
        model_factory=model_factory,
        client=bedrock_client,
        mode="code_with_stdin",
        reward_fn=reward_fn,
        system_prompt=SYSTEM_PROMPT,
        max_tool_iters=MAX_TOOL_ITERS,
        max_tool_calls=MAX_TOOL_CALLS,
        verbose=False,
    )

    # Create action with all test cases in ground truth (all hidden for reward)
    action = Action(
        message=prompt,
        task_context=TaskContext(
            ground_truth={
                "inputs": hidden_inputs,
                "outputs": hidden_outputs,
            },
            conversation_history=[],
        ),
    )

    # Execute environment step
    step_result = await env.step(action)

    # Extract token data from observation
    token_obs = step_result.observation.tokens
    sample.tokens = token_obs.token_ids
    sample.loss_mask = token_obs.rollout_loss_mask
    sample.rollout_log_probs = token_obs.rollout_logprobs
    sample.response_length = len(token_obs.rollout_token_ids)
    sample.response = state.tokenizer.decode(token_obs.rollout_token_ids, skip_special_tokens=False)

    # Set status
    if step_result.termination_reason.value == "task_complete":
        sample.status = Sample.Status.COMPLETED
    else:
        sample.status = Sample.Status.TRUNCATED

    # Set step result for custom rollout logging in `log_rollout_metrics`
    sample.step_result = step_result

    # Cleanup
    await env.cleanup()
    await reward_fn.cleanup()

    # Compute final reward with shaping
    base_reward = step_result.reward.reward  # Proportion of hidden tests passed [0.0, 1.0]

    if base_reward == 1.0:
        # Perfect solution - full reward
        sample.reward = 1.0
    elif base_reward > 0:
        # Partial credit: scale partial rewards to encourage progress
        # Reward in [0.4, 0.9] for partial success
        sample.reward = 0.4 + (base_reward * 0.5)
    else:
        # Failed all tests - give small credit for tool usage to encourage exploration
        tool_iters = sample.metrics.get("tool_iters", 0)
        # Penalty in [-0.8, -0.5] based on tool usage
        sample.reward = min(-0.5, -0.8 + (tool_iters / MAX_TOOL_ITERS) * 0.3)

    return sample
