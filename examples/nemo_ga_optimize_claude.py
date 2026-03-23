"""
Full Agent Optimization: Use NeMo Agent Toolkit's optimizer to tune
every aspect of a Claude agent — model, tools, prompts, retrieval, and params.

This shows how to define a NeMo workflow with OptimizableFields that cover
the entire Claude agent configuration, then run `nat optimize` to find the
best combination.

Prerequisites:
    pip install nvidia-nat claude-agent-sdk
    export ANTHROPIC_API_KEY=your_key
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import query, ClaudeSDKClient, ClaudeAgentOptions


# ============================================================================
# 1. Define optimizable agent configuration (NeMo OptimizableField pattern)
# ============================================================================

# In a real NeMo workflow, these would be OptimizableField declarations:
#
# from nvidia_nat import OptimizableField, SearchSpace, WorkflowConfig
#
# class ClaudeAgentConfig(WorkflowConfig):
#     # --- GA-optimized (prompts) ---
#     system_prompt: str = OptimizableField(
#         default="You are a helpful coding assistant.",
#         space=SearchSpace(is_prompt=True),
#     )
#
#     # --- Optuna-optimized (categorical) ---
#     model_name: str = OptimizableField(
#         default="claude-sonnet-4-6",
#         space=SearchSpace(values=[
#             "claude-haiku-4-5-20251001",
#             "claude-sonnet-4-6",
#             "claude-opus-4-6",
#         ]),
#     )
#     tool_set: str = OptimizableField(
#         default="standard",
#         space=SearchSpace(values=["minimal", "standard", "full"]),
#     )
#     retrieval_strategy: str = OptimizableField(
#         default="semantic",
#         space=SearchSpace(values=["semantic", "hybrid", "bm25"]),
#     )
#
#     # --- Optuna-optimized (numerical) ---
#     temperature: float = OptimizableField(
#         default=1.0,
#         space=SearchSpace(low=0.0, high=1.0),
#     )
#     top_p: float = OptimizableField(
#         default=1.0,
#         space=SearchSpace(low=0.1, high=1.0),
#     )
#     max_tokens: int = OptimizableField(
#         default=1024,
#         space=SearchSpace(low=256, high=4096, step=256),
#     )
#     max_turns: int = OptimizableField(
#         default=3,
#         space=SearchSpace(low=1, high=10),
#     )
#     top_k_documents: int = OptimizableField(
#         default=5,
#         space=SearchSpace(low=1, high=20),
#     )


# For this demo, we use a plain dataclass to represent the agent config.
@dataclass
class AgentConfig:
    system_prompt: str
    model_name: str
    tool_set: str
    temperature: float
    max_turns: int
    top_k_documents: int


# Tool set definitions
TOOL_SETS = {
    "minimal": ["Read"],
    "standard": ["Read", "Write", "Bash"],
    "full": ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
}


# ============================================================================
# 2. Claude agent runner — called by NeMo for each optimization trial
# ============================================================================

async def run_claude_agent(config: AgentConfig, user_query: str) -> dict[str, Any]:
    """Run a Claude agent with the given configuration.

    NeMo's optimizer calls this for every trial, varying the config params.
    Returns output + metadata for evaluation.
    """
    tools = TOOL_SETS.get(config.tool_set, TOOL_SETS["standard"])

    result_parts = []
    tool_calls = 0

    async for message in query(
        prompt=user_query,
        system_prompt=config.system_prompt,
        model=config.model_name,
        max_turns=config.max_turns,
        options={
            "temperature": config.temperature,
            "allowed_tools": tools,
        },
    ):
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    result_parts.append(block.text)
                elif hasattr(block, "tool_use"):
                    tool_calls += 1

    output_text = "\n".join(result_parts)

    return {
        "output": output_text,
        "tool_calls": tool_calls,
        "model": config.model_name,
        "token_estimate": len(output_text.split()) * 1.3,  # rough
    }


# ============================================================================
# 3. Evaluation — NeMo's eval system scores each trial
# ============================================================================

# Cost per 1K output tokens (approximate, USD)
MODEL_COSTS = {
    "claude-haiku-4-5-20251001": 0.001,
    "claude-sonnet-4-6": 0.015,
    "claude-opus-4-6": 0.075,
}


@dataclass
class TrialScore:
    correctness: float  # 0-1
    cost: float         # USD estimate
    latency: float      # seconds (simulated)
    tool_efficiency: float  # 0-1


def evaluate_trial(result: dict, expected: str) -> TrialScore:
    """Score a single trial. NeMo's eval_metrics handles this in production."""
    output = result["output"]
    model = result["model"]

    correctness = 1.0 if expected.lower() in output.lower() else 0.0

    tokens = result["token_estimate"]
    cost_per_1k = MODEL_COSTS.get(model, 0.015)
    cost = tokens / 1000 * cost_per_1k

    # Simulated latency — in production, NeMo's profiler measures real latency
    latency_multiplier = {"claude-haiku-4-5-20251001": 0.5, "claude-sonnet-4-6": 1.0, "claude-opus-4-6": 2.5}
    latency = tokens * 0.01 * latency_multiplier.get(model, 1.0)

    tool_calls = result["tool_calls"]
    tool_efficiency = max(0.0, 1.0 - tool_calls / 10)

    return TrialScore(
        correctness=correctness,
        cost=cost,
        latency=latency,
        tool_efficiency=tool_efficiency,
    )


# ============================================================================
# 4. Demo: Simulate what `nat optimize` does across the full search space
# ============================================================================

EVAL_DATASET = [
    {"query": "How do I reverse a list in Python?", "expected": "[::-1]"},
    {"query": "What is a context manager in Python?", "expected": "with"},
    {"query": "How do I handle exceptions in Python?", "expected": "try"},
]

# Search space — NeMo's optimizer would explore this automatically
SEARCH_SPACE = {
    "models": [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ],
    "tool_sets": ["minimal", "standard", "full"],
    "temperatures": [0.0, 0.3, 0.7, 1.0],
    "prompts": [
        "You are a helpful coding assistant. Be concise.",
        "You are a Python expert. Answer in one paragraph with code.",
        "Answer directly with a code example. No preamble.",
        "You are a senior developer. Practical, short answers only.",
    ],
}


async def demo_full_optimization():
    """Simulate NeMo's full agent optimization loop.

    In production, you'd just run:
        nat optimize --config_file workflow.yaml

    NeMo handles the full Optuna + GA loop, experiment tracking,
    and Pareto front computation automatically.
    """
    print("=" * 70)
    print("Full Agent Optimization: Model + Tools + Prompts + Params")
    print("=" * 70)
    print()
    print("In production: configure OptimizableFields + YAML, run `nat optimize`")
    print("This demo shows the concept with a simplified search.\n")

    # Simulate a few trials across the search space
    trials = [
        AgentConfig("Be concise.", "claude-haiku-4-5-20251001", "minimal", 0.0, 1, 3),
        AgentConfig("Be concise.", "claude-sonnet-4-6", "standard", 0.3, 3, 5),
        AgentConfig("Answer with code.", "claude-opus-4-6", "full", 0.7, 5, 10),
        AgentConfig("Be concise.", "claude-sonnet-4-6", "minimal", 0.0, 1, 3),
    ]

    results = []

    for i, config in enumerate(trials):
        print(f"Trial {i+1}: model={config.model_name.split('-')[1]}, "
              f"tools={config.tool_set}, temp={config.temperature}")

        trial_scores = []
        for item in EVAL_DATASET:
            result = await run_claude_agent(config, item["query"])
            score = evaluate_trial(result, item["expected"])
            trial_scores.append(score)

        avg = TrialScore(
            correctness=sum(s.correctness for s in trial_scores) / len(trial_scores),
            cost=sum(s.cost for s in trial_scores) / len(trial_scores),
            latency=sum(s.latency for s in trial_scores) / len(trial_scores),
            tool_efficiency=sum(s.tool_efficiency for s in trial_scores) / len(trial_scores),
        )

        # Weighted composite score (NeMo supports multi-objective Pareto too)
        composite = (
            avg.correctness * 0.5
            + (1 - min(avg.cost / 0.01, 1.0)) * 0.3  # normalize cost
            + avg.tool_efficiency * 0.2
        )

        print(f"  → correctness={avg.correctness:.2f}  cost=${avg.cost:.4f}  "
              f"composite={composite:.3f}")
        results.append((config, composite, avg))

    # Find best
    results.sort(key=lambda x: x[1], reverse=True)
    best_config, best_score, best_avg = results[0]

    print(f"\n{'=' * 70}")
    print(f"BEST CONFIG (composite={best_score:.3f}):")
    print(f"  Model:       {best_config.model_name}")
    print(f"  Tools:       {best_config.tool_set} → {TOOL_SETS[best_config.tool_set]}")
    print(f"  Temperature: {best_config.temperature}")
    print(f"  Max turns:   {best_config.max_turns}")
    print(f"  Prompt:      {best_config.system_prompt}")
    print(f"  Correctness: {best_avg.correctness:.2f}")
    print(f"  Avg cost:    ${best_avg.cost:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(demo_full_optimization())
