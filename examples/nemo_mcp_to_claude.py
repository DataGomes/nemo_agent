"""
Pattern 1: NeMo Agent Toolkit workflows exposed as MCP tools for Claude Agent SDK.

This example shows how to:
1. Publish a NeMo workflow as an MCP server
2. Connect it to Claude Agent SDK as an external MCP tool
3. Let Claude orchestrate NeMo-powered workflows

Prerequisites:
    pip install nvidia-nat claude-agent-sdk
"""

import anyio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions


async def main():
    # Configure Claude Agent SDK to use a NeMo MCP server as a tool.
    #
    # In production, the NeMo workflow would be published via:
    #   nvidia-nat serve --workflow my_workflow --transport stdio
    #
    # Here we connect to it as an external MCP server.
    options = ClaudeAgentOptions(
        system_prompt=(
            "You have access to NeMo Agent Toolkit workflows via MCP tools. "
            "Use the nemo_workflow tools when the user asks for data analysis, "
            "model inference, or GPU-accelerated tasks."
        ),
        mcp_servers={
            # External NeMo MCP server (stdio transport)
            "nemo_workflow": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "nvidia_nat.serve", "--workflow", "example_workflow"],
            }
        },
        # Auto-approve NeMo workflow tools
        allowed_tools=["mcp__nemo_workflow__*"],
        max_turns=5,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Analyze the performance metrics from the last training run")

        async for message in client.receive_response():
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        print(block.text)


if __name__ == "__main__":
    anyio.run(main)
