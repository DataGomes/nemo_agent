# NeMo Agent Toolkit + Claude Agent SDK Compatibility Test

This repo tests whether NVIDIA's NeMo Agent Toolkit (`nvidia-nat`) and Anthropic's
Claude Agent SDK (`claude-agent-sdk`) can be used together.

## TL;DR

**Yes, they can be used together.** The primary integration path is the
**Model Context Protocol (MCP)**, which both projects support natively.

## Compatibility Summary

| Aspect                  | NeMo Agent Toolkit        | Claude Agent SDK          | Compatible? |
|-------------------------|---------------------------|---------------------------|-------------|
| Python version          | 3.11, 3.12, 3.13         | 3.10+                     | Yes (3.11+) |
| MCP server support      | Yes (FastMCP publishing)  | Yes (in-process + external)| Yes         |
| MCP client support      | Yes                       | Yes                       | Yes         |
| A2A protocol            | Yes                       | Not natively              | Partial     |
| Async runtime           | asyncio                   | anyio (asyncio-compatible)| Yes         |
| Package conflicts       | None known                | None known                | Yes         |

## Integration Patterns

### Pattern 1: NeMo workflows as MCP tools for Claude (Recommended)
Publish NeMo Agent Toolkit workflows as MCP servers, then connect them
as external MCP tools to the Claude Agent SDK.

### Pattern 2: Claude as a custom tool inside NeMo workflows
Use Claude Agent SDK's `query()` function inside a NeMo Agent Toolkit
custom tool or workflow node.

### Pattern 3: NeMo as observability + evaluation layer for Claude agents
Wrap Claude Agent SDK calls in NeMo workflows to get enterprise-grade tracing,
evaluation, and experiment tracking via LangSmith or OpenTelemetry.

### Pattern 4: Genetic Algorithm optimization for Claude prompts
Use NeMo's built-in GA optimizer to evolve better system prompts, tool
descriptions, and hyperparameters (temperature, max_turns) for Claude agents.

### Pattern 5: Shared MCP tool ecosystem
Both frameworks can consume the same MCP tool servers, enabling a shared
tool layer across heterogeneous agent architectures.

## Quick Start

```bash
pip install -r requirements.txt
python test_compatibility.py
```

## Files

- `requirements.txt` — Combined dependencies
- `test_compatibility.py` — Import and API compatibility checks
- `examples/nemo_mcp_to_claude.py` — Pattern 1 example
- `examples/claude_tool_in_nemo.py` — Pattern 2 example
- `examples/nemo_ga_optimize_claude.py` — Pattern 4: GA prompt optimization
- `ANALYSIS.md` — Detailed compatibility analysis
