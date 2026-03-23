# Detailed Compatibility Analysis: NeMo Agent Toolkit + Claude Agent SDK

## 1. Python Version Overlap

- **NeMo Agent Toolkit**: Requires Python 3.11, 3.12, or 3.13
- **Claude Agent SDK**: Requires Python 3.10+
- **Overlap**: Python 3.11, 3.12, 3.13 — use any of these.

## 2. MCP Protocol — The Key Integration Point

Both frameworks have first-class MCP support, making this the natural bridge.

### NeMo as MCP Server → Claude as MCP Client

NeMo Agent Toolkit can publish any workflow as an MCP server via its FastMCP
runtime. Claude Agent SDK can consume external MCP servers through the
`mcp_servers` option in `ClaudeAgentOptions`.

```python
# Claude Agent SDK consuming a NeMo MCP server
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

options = ClaudeAgentOptions(
    mcp_servers={
        "nemo_workflow": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "nvidia_nat.serve", "--workflow", "my_workflow"]
        }
    },
    allowed_tools=["mcp__nemo_workflow__*"]
)
```

### Claude as MCP Server → NeMo as MCP Client

The Claude Agent SDK can create in-process MCP servers with `create_sdk_mcp_server()`.
These could be exposed to NeMo workflows via the MCP protocol.

## 3. A2A Protocol

NeMo supports the Agent-to-Agent (A2A) protocol for distributed multi-agent
systems. Claude Agent SDK does not natively support A2A. However, a Claude
agent could be wrapped in an A2A-compatible server to participate in NeMo's
multi-agent orchestration.

## 4. Async Runtime Compatibility

- **NeMo**: Standard `asyncio`
- **Claude Agent SDK**: Uses `anyio` which is compatible with `asyncio`
- No conflicts expected — both can run in the same event loop.

## 5. Dependency Conflicts

Both packages use standard Python dependencies. Key shared areas:
- `httpx` — Used by both, version ranges are compatible
- `pydantic` — Both use Pydantic v2
- MCP libraries — Both use the standard `mcp` package

No known dependency conflicts.

## 6. Integration Patterns (Detailed)

### Pattern 1: NeMo Workflows as Claude Tools (Recommended)

**Use case**: You have optimized NeMo Agent Toolkit workflows (with profiling,
observability, RL-tuned models) and want Claude to orchestrate them.

**How it works**:
1. Build your workflow in NeMo Agent Toolkit
2. Publish it as an MCP server using FastMCP
3. Connect it to Claude Agent SDK as an external MCP server
4. Claude can now invoke NeMo workflows as tools

**Benefits**:
- NeMo handles performance optimization (Agent Performance Primitives)
- Claude handles natural language reasoning and orchestration
- Clean separation of concerns via MCP

### Pattern 2: Claude as a Tool Inside NeMo

**Use case**: You have a NeMo multi-agent system and want to add Claude's
reasoning capabilities as one of the agents/tools.

**How it works**:
1. Create a NeMo custom tool that calls `claude_agent_sdk.query()`
2. The tool sends prompts to Claude and returns results
3. NeMo workflows can invoke Claude for tasks needing strong reasoning

### Pattern 3: Hybrid Multi-Agent Architecture

**Use case**: Complex systems with both NVIDIA NIM models and Claude models.

**How it works**:
1. NeMo orchestrates NVIDIA NIM-powered agents for specialized tasks
2. Claude Agent SDK runs Claude-powered agents for reasoning-heavy tasks
3. Both share MCP tool servers for common capabilities (file I/O, APIs, etc.)
4. Optionally connect via A2A for cross-framework agent communication

## 7. Limitations & Caveats

1. **No native integration**: Neither project explicitly supports the other.
   Integration relies on the shared MCP protocol.
2. **A2A gap**: Claude Agent SDK lacks native A2A support, requiring a wrapper.
3. **Auth model differences**: NeMo uses NVIDIA API keys; Claude uses Anthropic
   API keys. Both need to be configured when using both frameworks.
4. **Observability**: NeMo's LangSmith integration won't automatically trace
   Claude Agent SDK calls. You'd need separate observability for each.
