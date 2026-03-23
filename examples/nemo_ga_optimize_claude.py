"""
Pattern 3: Use NeMo Agent Toolkit's Genetic Algorithm to optimize
Claude Agent SDK system prompts and hyperparameters.

This example shows the architecture for using NeMo's optimizer to
evolve better prompts for Claude-powered agents.

Prerequisites:
    pip install nvidia-nat claude-agent-sdk
    export ANTHROPIC_API_KEY=your_key
"""

import asyncio
from dataclasses import dataclass

from claude_agent_sdk import query


# ---------------------------------------------------------------------------
# 1. Define the Claude agent workflow that NeMo will optimize
# ---------------------------------------------------------------------------

async def run_claude_agent(
    system_prompt: str,
    user_query: str,
    temperature: float = 1.0,
    max_turns: int = 3,
) -> str:
    """Run a Claude agent with the given parameters.

    NeMo's optimizer will call this with different system_prompt and
    temperature values across generations/trials.
    """
    result_parts = []
    async for message in query(
        prompt=user_query,
        system_prompt=system_prompt,
        max_turns=max_turns,
        options={"temperature": temperature},
    ):
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    result_parts.append(block.text)

    return "\n".join(result_parts)


# ---------------------------------------------------------------------------
# 2. Define evaluation metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    correctness: float  # 0.0 - 1.0
    conciseness: float  # 0.0 - 1.0
    cost: float         # estimated token cost


async def evaluate_output(output: str, expected: str) -> EvalResult:
    """Score a Claude agent output. In production, use NeMo's eval system."""
    # Simplified scoring — NeMo's eval_metrics would handle this via
    # LLM judges, exact match, ROUGE, or custom evaluators.
    correctness = 1.0 if expected.lower() in output.lower() else 0.0
    conciseness = max(0.0, 1.0 - len(output) / 2000)
    cost = len(output) * 0.00001  # rough estimate
    return EvalResult(
        correctness=correctness,
        conciseness=conciseness,
        cost=cost,
    )


# ---------------------------------------------------------------------------
# 3. GA optimization loop (conceptual — NeMo handles this automatically)
# ---------------------------------------------------------------------------

# In practice, you'd configure this in a NeMo workflow YAML:
#
# optimizer:
#   ga_generations: 10
#   ga_population_size: 8
#   n_trials_numeric: 20
#   reps_per_param_set: 3
#   parameters:
#     system_prompt:
#       type: prompt
#       prompt_purpose: "Instruct Claude to answer coding questions concisely"
#       initial_value: "You are a helpful coding assistant."
#     temperature:
#       type: float
#       low: 0.0
#       high: 1.0
#   eval_metrics:
#     - name: correctness
#       type: llm_judge
#     - name: conciseness
#       type: custom
#     - name: cost
#       type: token_count

EVAL_DATASET = [
    {
        "query": "How do I reverse a list in Python?",
        "expected": "[::-1]",
    },
    {
        "query": "What is a context manager in Python?",
        "expected": "with",
    },
    {
        "query": "How do I handle exceptions in Python?",
        "expected": "try",
    },
]

INITIAL_PROMPTS = [
    "You are a helpful coding assistant. Be concise.",
    "You are a Python expert. Answer in one paragraph or less.",
    "Answer coding questions directly with code examples. Be brief.",
    "You are a senior developer. Give practical, short answers.",
]


async def run_optimization_demo():
    """Demonstrate the GA optimization concept.

    In production, NeMo Agent Toolkit handles the full GA loop:
    - Population management
    - LLM-powered mutation
    - Recombination
    - Optuna integration for numeric params
    - Parallel evaluation
    - Experiment tracking via LangSmith
    """
    print("=" * 60)
    print("GA Prompt Optimization for Claude Agents (Demo)")
    print("=" * 60)
    print()
    print("This demo shows the concept. In production, configure")
    print("this in a NeMo workflow YAML and run: nat optimize")
    print()

    best_prompt = None
    best_score = -1.0

    for gen in range(2):  # NeMo would run ga_generations
        print(f"--- Generation {gen + 1} ---")

        for i, prompt in enumerate(INITIAL_PROMPTS):
            total_score = 0.0

            for item in EVAL_DATASET:
                output = await run_claude_agent(
                    system_prompt=prompt,
                    user_query=item["query"],
                    temperature=0.3,
                    max_turns=1,
                )
                result = await evaluate_output(output, item["expected"])
                total_score += result.correctness + result.conciseness

            avg_score = total_score / len(EVAL_DATASET)
            print(f"  Prompt {i}: avg_score={avg_score:.3f}")

            if avg_score > best_score:
                best_score = avg_score
                best_prompt = prompt

        print(f"  Best so far: score={best_score:.3f}")
        print()

        # In NeMo, the GA would now mutate/recombine prompts for next gen.
        # The LLM-powered mutation creates new prompt variants.

    print("=" * 60)
    print(f"Best prompt (score={best_score:.3f}):")
    print(f"  {best_prompt}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_optimization_demo())
