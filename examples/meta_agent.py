"""
Meta-Agent: A Claude agent that optimizes other Claude agents using NeMo.

The user describes what they want in natural language. The meta-agent:
1. Generates the NeMo optimizer config (search space, eval metrics, weights)
2. Creates an evaluation dataset
3. Runs `nat optimize`
4. Interprets results and picks the best config from the Pareto front
5. Deploys the optimized agent
6. Monitors and re-optimizes when performance drifts

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  User: "I need a code review agent. Accuracy matters     │
    │         most but keep cost reasonable."                   │
    └──────────────────────┬───────────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────────┐
    │  META-AGENT (Claude Agent SDK)                           │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │ Step 1: Understand the task                        │  │
    │  │  - Parse natural language intent                   │  │
    │  │  - Identify domain, constraints, priorities        │  │
    │  └───────────────────┬────────────────────────────────┘  │
    │  ┌───────────────────▼────────────────────────────────┐  │
    │  │ Step 2: Generate NeMo optimizer config             │  │
    │  │  - Choose search space (models, tools, params)     │  │
    │  │  - Set eval metric weights from user intent        │  │
    │  │  - Write YAML config file                          │  │
    │  └───────────────────┬────────────────────────────────┘  │
    │  ┌───────────────────▼────────────────────────────────┐  │
    │  │ Step 3: Generate evaluation dataset                │  │
    │  │  - Create test cases relevant to the domain        │  │
    │  │  - Include edge cases, adversarial inputs          │  │
    │  │  - Write expected outputs / rubrics                │  │
    │  └───────────────────┬────────────────────────────────┘  │
    │  ┌───────────────────▼────────────────────────────────┐  │
    │  │ Step 4: Run NeMo optimization                      │  │
    │  │  - Execute: nat optimize --config generated.yaml   │  │
    │  │  - Monitor progress, handle errors                 │  │
    │  └───────────────────┬────────────────────────────────┘  │
    │  ┌───────────────────▼────────────────────────────────┐  │
    │  │ Step 5: Interpret results                          │  │
    │  │  - Analyze Pareto front                            │  │
    │  │  - Pick best config matching user's priorities     │  │
    │  │  - Explain tradeoffs in plain language             │  │
    │  └───────────────────┬────────────────────────────────┘  │
    │  ┌───────────────────▼────────────────────────────────┐  │
    │  │ Step 6: Deploy optimized agent                     │  │
    │  │  - Write final ClaudeAgentOptions config           │  │
    │  │  - Optionally: set up monitoring for drift         │  │
    │  └────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────┘

Prerequisites:
    pip install nvidia-nat claude-agent-sdk
    export ANTHROPIC_API_KEY=your_key
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from textwrap import dedent

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    query,
    tool,
    create_sdk_mcp_server,
)


# ============================================================================
# MCP Tools for the Meta-Agent
# ============================================================================
# These tools let the meta-agent interact with NeMo and the filesystem.

@tool(
    "generate_optimizer_config",
    "Generate a NeMo Agent Toolkit optimizer YAML config file",
    {
        "config_yaml": str,  # The YAML content to write
        "output_path": str,  # Where to save it
    },
)
async def generate_optimizer_config(args: dict) -> dict:
    """Write a NeMo optimizer config YAML to disk."""
    output_path = Path(args["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(args["config_yaml"])
    return {
        "content": [
            {"type": "text", "text": f"Config written to {output_path}"}
        ]
    }


@tool(
    "generate_eval_dataset",
    "Generate an evaluation dataset for the optimizer",
    {
        "dataset_json": str,  # JSON array of eval examples
        "output_path": str,
    },
)
async def generate_eval_dataset(args: dict) -> dict:
    """Write an eval dataset to disk."""
    output_path = Path(args["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Validate it's valid JSON
    data = json.loads(args["dataset_json"])
    output_path.write_text(json.dumps(data, indent=2))
    return {
        "content": [
            {
                "type": "text",
                "text": f"Dataset with {len(data)} examples written to {output_path}",
            }
        ]
    }


@tool(
    "run_nemo_optimizer",
    "Run the NeMo Agent Toolkit optimizer with a config file",
    {
        "config_path": str,
    },
)
async def run_nemo_optimizer(args: dict) -> dict:
    """Execute `nat optimize` and return results."""
    config_path = args["config_path"]

    proc = await asyncio.create_subprocess_exec(
        "nat", "optimize", "--config_file", config_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Optimizer failed (exit {proc.returncode}):\n{stderr.decode()}",
                }
            ]
        }

    return {
        "content": [
            {"type": "text", "text": f"Optimization complete:\n{stdout.decode()}"}
        ]
    }


@tool(
    "read_optimization_results",
    "Read and parse optimization results from NeMo output directory",
    {
        "results_dir": str,
    },
)
async def read_optimization_results(args: dict) -> dict:
    """Read the optimizer output files and return parsed results."""
    results_dir = Path(args["results_dir"])

    results = {}
    for f in results_dir.glob("*.json"):
        results[f.name] = json.loads(f.read_text())

    return {
        "content": [
            {"type": "text", "text": json.dumps(results, indent=2)}
        ]
    }


@tool(
    "deploy_optimized_agent",
    "Write the final optimized agent configuration as a Python file",
    {
        "agent_code": str,
        "output_path": str,
    },
)
async def deploy_optimized_agent(args: dict) -> dict:
    """Write the optimized agent to a Python file."""
    output_path = Path(args["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(args["agent_code"])
    return {
        "content": [
            {"type": "text", "text": f"Optimized agent deployed to {output_path}"}
        ]
    }


# ============================================================================
# The Meta-Agent
# ============================================================================

META_AGENT_SYSTEM_PROMPT = dedent("""\
    You are a META-AGENT: an AI agent optimizer. Your job is to take a user's
    description of what they want an AI agent to do, and then:

    1. UNDERSTAND: Parse the user's intent — what domain, what task, what
       matters most (accuracy? cost? speed? safety?).

    2. CONFIGURE: Generate a NeMo Agent Toolkit optimizer config that defines:
       - Search space: which Claude models (haiku/sonnet/opus), tool sets,
         temperatures, max_turns, retrieval strategies to explore
       - Eval metrics: correctness, cost, latency, safety — with weights
         derived from the user's stated priorities
       - GA settings: population size, generations for prompt evolution

    3. EVALUATE: Generate a high-quality evaluation dataset for the domain:
       - 10-20 representative test cases
       - Edge cases and adversarial inputs
       - Expected outputs or scoring rubrics

    4. OPTIMIZE: Run the NeMo optimizer (`nat optimize`) on the config.

    5. INTERPRET: Analyze the Pareto front / best configs and explain the
       results to the user in plain language. Show tradeoffs.

    6. DEPLOY: Write the final optimized agent configuration as ready-to-use
       Python code using ClaudeAgentOptions.

    Priority inference rules:
    - "accurate", "correct", "reliable" → high correctness weight
    - "cheap", "affordable", "budget" → high cost weight
    - "fast", "real-time", "low latency" → high latency weight
    - "safe", "secure", "compliant" → add safety evaluator
    - If no priority stated, use balanced weights

    Model selection guidance:
    - Always include all three Claude tiers in the search space
    - The optimizer will find which one is best for the specific task
    - Don't pre-judge — Haiku with a good prompt often beats Opus with a bad one

    Tool selection guidance:
    - Define 3-4 tool set variations (minimal, standard, full, domain-specific)
    - Fewer tools often improves accuracy (less distraction)
    - More tools needed for complex multi-step tasks

    You have tools to: generate configs, create datasets, run the optimizer,
    read results, and deploy the final agent. Use them in sequence.
""")


async def run_meta_agent(user_request: str, work_dir: str = "/tmp/meta_agent"):
    """Run the meta-agent to optimize a Claude agent based on user request."""

    # Create the MCP server with all meta-agent tools
    meta_tools_server = create_sdk_mcp_server(
        name="meta-agent-tools",
        version="1.0.0",
        tools=[
            generate_optimizer_config,
            generate_eval_dataset,
            run_nemo_optimizer,
            read_optimization_results,
            deploy_optimized_agent,
        ],
    )

    options = ClaudeAgentOptions(
        system_prompt=META_AGENT_SYSTEM_PROMPT,
        mcp_servers={
            "meta": meta_tools_server,
        },
        allowed_tools=[
            "mcp__meta__generate_optimizer_config",
            "mcp__meta__generate_eval_dataset",
            "mcp__meta__run_nemo_optimizer",
            "mcp__meta__read_optimization_results",
            "mcp__meta__deploy_optimized_agent",
        ],
        max_turns=15,  # Enough for the full optimization pipeline
    )

    print("=" * 70)
    print("META-AGENT: Claude optimizing Claude via NeMo")
    print("=" * 70)
    print(f"\nUser request: {user_request}\n")

    async with ClaudeSDKClient(options=options) as client:
        prompt = dedent(f"""\
            The user wants you to create and optimize an AI agent for them.
            Here is their request:

            "{user_request}"

            Work directory: {work_dir}

            Please:
            1. Analyze their requirements and priorities
            2. Generate a NeMo optimizer config at {work_dir}/optimizer.yaml
            3. Generate an eval dataset at {work_dir}/eval_dataset.json
            4. Run the optimizer
            5. Interpret the results and pick the best config
            6. Deploy the optimized agent to {work_dir}/optimized_agent.py
            7. Explain what you chose and why
        """)

        await client.query(prompt)

        async for message in client.receive_response():
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        print(block.text)


# ============================================================================
# Example: Self-Optimizing Agent Loop
# ============================================================================

async def self_optimizing_loop(
    agent_config_path: str,
    eval_dataset_path: str,
    check_interval_hours: int = 24,
):
    """A Claude agent that monitors its own performance and re-optimizes.

    This is the most 'meta' version: the agent watches its own metrics
    and triggers NeMo re-optimization when performance degrades.

    1. Run the deployed agent normally, collecting metrics
    2. Periodically, the meta-agent checks if metrics have drifted
    3. If drift detected → re-run NeMo optimization with updated data
    4. Hot-swap the agent config with the new optimized version
    """

    meta_prompt = dedent(f"""\
        You are monitoring a deployed Claude agent. Check its recent
        performance metrics and decide if re-optimization is needed.

        Current config: {agent_config_path}
        Eval dataset: {eval_dataset_path}

        Check these signals:
        - Has average correctness dropped below the baseline?
        - Has cost increased (model pricing changes, longer outputs)?
        - Has latency increased?
        - Are there new failure modes in recent logs?

        If any metric has degraded by more than 10% from baseline:
        1. Analyze what changed (data distribution shift? new edge cases?)
        2. Update the eval dataset with recent failure cases
        3. Re-run NeMo optimization
        4. Deploy the new config if it's better than current

        If metrics are stable, report "No optimization needed" and
        summarize current performance.
    """)

    print(f"Self-optimizing loop: checking every {check_interval_hours}h")
    print("(In production, this would run as a cron job or scheduled task)\n")

    # Single check for demo purposes
    async for message in query(prompt=meta_prompt, max_turns=10):
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text)


# ============================================================================
# Entry point
# ============================================================================

EXAMPLE_REQUESTS = [
    (
        "I need a code review agent. Accuracy is critical — it needs to catch "
        "real bugs, not hallucinate fake ones. Cost should be reasonable but "
        "I'll pay more for accuracy. It should work with Python and TypeScript."
    ),
    (
        "Build me a customer support agent that answers billing questions. "
        "It needs to be fast and cheap — we get 10K queries/day. Accuracy "
        "should be good enough but doesn't need to be perfect."
    ),
    (
        "I want a research agent that summarizes academic papers. It must be "
        "very accurate and cite sources. Cost and latency don't matter."
    ),
]


async def main():
    print("Select an example request (or type your own):\n")
    for i, req in enumerate(EXAMPLE_REQUESTS, 1):
        print(f"  {i}. {req[:80]}...")
    print(f"  {len(EXAMPLE_REQUESTS) + 1}. Enter custom request")
    print()

    choice = input("Choice: ").strip()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(EXAMPLE_REQUESTS):
            request = EXAMPLE_REQUESTS[idx]
        else:
            request = input("Enter your request: ").strip()
    except ValueError:
        request = choice  # Treat as custom input

    work_dir = tempfile.mkdtemp(prefix="meta_agent_")
    print(f"Work directory: {work_dir}\n")

    await run_meta_agent(request, work_dir)


if __name__ == "__main__":
    asyncio.run(main())
