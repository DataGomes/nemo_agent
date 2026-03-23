"""
Pattern 2: Claude Agent SDK used as a tool inside a NeMo Agent Toolkit workflow.

This example shows how to:
1. Create a NeMo custom tool that calls Claude for complex reasoning
2. Use it within a NeMo workflow alongside NIM-powered agents

Prerequisites:
    pip install nvidia-nat claude-agent-sdk
    export ANTHROPIC_API_KEY=your_key
    export NVIDIA_API_KEY=your_key
"""

import asyncio
from claude_agent_sdk import query


async def claude_reasoning_tool(prompt: str) -> str:
    """Call Claude Agent SDK for complex reasoning tasks.

    This function can be registered as a custom tool in NeMo Agent Toolkit
    workflows, giving NeMo agents access to Claude's reasoning capabilities.
    """
    result_parts = []
    async for message in query(
        prompt=prompt,
        max_turns=1,
    ):
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    result_parts.append(block.text)

    return "\n".join(result_parts)


# Example: Register as a NeMo tool (conceptual — actual API depends on
# NeMo Agent Toolkit workflow configuration)
#
# from nvidia_nat import Workflow, Tool
#
# @workflow.tool("claude_reason")
# async def reason(query: str) -> str:
#     """Use Claude for multi-step reasoning."""
#     return await claude_reasoning_tool(query)
#
# workflow = Workflow(
#     name="hybrid_agents",
#     tools=[reason],
#     llm="nvidia/nemotron-3-nano-30b-a3b",  # NIM for fast tasks
# )


async def main():
    # Standalone demo — call Claude for a reasoning task
    print("Calling Claude Agent SDK for reasoning...\n")
    result = await claude_reasoning_tool(
        "Explain three ways GPU-accelerated inference differs from CPU inference"
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
