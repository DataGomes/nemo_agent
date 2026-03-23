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

## 7. NeMo Observability + Evaluation for Claude Agents

### The Idea

Use NeMo Agent Toolkit as the **observability, evaluation, and optimization
layer** on top of Claude Agent SDK workloads. NeMo provides enterprise-grade
instrumentation that Claude Agent SDK lacks natively.

### Observability

NeMo's observability system is plugin-based and event-driven. It traces every
step of agent workflows and exports telemetry to:
- **LangSmith** (native integration)
- **Langfuse** (also has a Claude Agent SDK integration via OpenTelemetry)
- **Phoenix**, **Weave**, or any **OpenTelemetry-compatible** backend

**How to apply to Claude agents**: Wrap Claude Agent SDK calls inside a NeMo
workflow. The NeMo workflow acts as the instrumented orchestrator — every call
to Claude, every tool use, every token gets traced. You get per-request traces,
token usage metrics, latency breakdowns, and cost tracking.

```python
# Conceptual: Claude calls inside a NeMo-traced workflow
from nvidia_nat import Workflow, step

@step(name="claude_reasoning")
async def reason_with_claude(question: str) -> str:
    """NeMo traces this step — timing, tokens, cost."""
    result = []
    async for msg in query(prompt=question, max_turns=1):
        if hasattr(msg, "content"):
            for block in msg.content:
                if hasattr(block, "text"):
                    result.append(block.text)
    return "\n".join(result)
```

**Shared OpenTelemetry**: Both NeMo and Langfuse's Claude Agent SDK integration
use OpenTelemetry. You can configure a single OTel collector that receives
spans from both, giving unified observability across the full stack.

### Evaluation

NeMo's `nat eval` system can evaluate Claude agent outputs:
- Define eval questions/datasets
- Run the Claude-powered workflow against them
- Score outputs with built-in or custom evaluators (correctness, latency, etc.)
- Track results as structured experiments in LangSmith

This is valuable for regression testing Claude agent behavior across prompt
changes, model version upgrades, or configuration tweaks.

## 8. Genetic Algorithm for Prompt Optimization

### The Idea

NeMo Agent Toolkit has a **dual optimization strategy**:
1. **Optuna** — optimizes numerical hyperparameters (temperature, top_p, max_tokens)
2. **Custom Genetic Algorithm (GA)** — optimizes prompts by evolving a population
   of prompt candidates over multiple generations

### How the GA Works

1. **Population**: Starts with a population of prompt candidates
2. **Evaluation**: Each prompt is run through the workflow and scored by evaluators
3. **Selection**: Best-performing prompts survive
4. **Mutation**: LLM-powered mutation generates new prompt variants
5. **Recombination** (optional): Combines elements of successful prompts
6. **Repeat** for `ga_generations` generations

### Applying This to Claude Agent SDK Workloads

This is where it gets interesting. You can use NeMo's GA to **automatically
optimize the system prompts, tool descriptions, and instructions** you feed
to Claude agents.

**Architecture**:

```
┌─────────────────────────────────────────────────────┐
│  NeMo Agent Toolkit (Optimizer Layer)               │
│                                                     │
│  GA Population: [prompt_v1, prompt_v2, ..., prompt_N]│
│       │                                             │
│       ▼                                             │
│  For each prompt candidate:                         │
│    ┌──────────────────────────────────┐             │
│    │ Claude Agent SDK                 │             │
│    │ - system_prompt = candidate      │             │
│    │ - Run against eval dataset       │             │
│    │ - Collect outputs                │             │
│    └──────────────────────────────────┘             │
│       │                                             │
│       ▼                                             │
│  Evaluators score each candidate                    │
│  GA selects → mutates → recombines → next gen       │
│                                                     │
│  Also optimized by Optuna:                          │
│    - temperature, top_p, max_tokens                 │
│    - max_turns, tool selection                      │
└─────────────────────────────────────────────────────┘
```

**What you can optimize**:
- System prompts for Claude agents
- Tool descriptions (how tools are presented to Claude)
- Few-shot examples within prompts
- Numerical params: temperature, top_p, max_tokens
- Workflow-level params: max_turns, tool allowlists

**Configuration** (conceptual NeMo optimizer config):

```yaml
optimizer:
  ga_generations: 10
  ga_population_size: 8
  n_trials_numeric: 20
  reps_per_param_set: 3

  parameters:
    system_prompt:
      type: prompt
      prompt_purpose: "Instruct a Claude agent to analyze code quality"
      initial_value: "You are a code review assistant..."

    temperature:
      type: float
      low: 0.0
      high: 1.0

    max_turns:
      type: int
      low: 1
      high: 10

  eval_metrics:
    - name: correctness
      type: llm_judge
    - name: latency
      type: builtin
    - name: cost
      type: token_count
```

### Caveats for Claude-Specific Optimization

1. **Cost**: Each GA generation × population size × reps = many Claude API calls.
   A run of 10 generations × 8 candidates × 3 reps = 240 calls per eval question.
   Use `claude-haiku-4-5-20251001` for optimization runs, then validate with Opus.

2. **Rate limits**: Anthropic API rate limits may throttle large optimization runs.
   Add retry/backoff logic in the workflow step that calls Claude.

3. **Caching**: NeMo's profiler tracks prompt-prefix overlap. Use this data to
   design prompts with shared prefixes for better prompt caching on the Anthropic
   API (reduces cost and latency).

4. **Non-determinism**: Claude's outputs vary even at temperature=0. The
   `reps_per_param_set` config accounts for this by averaging over multiple runs.

## 9. GRPO / Reinforcement Learning (Advanced)

Beyond prompt-level optimization, NeMo also supports **model-level RL** via
NeMo RL (GRPO, DPO, SFT). This wouldn't directly apply to Claude (you can't
fine-tune Claude's weights), but it's relevant if your architecture uses a
mix of open models (via NIM) and Claude:

- Use GRPO to fine-tune a Nemotron model for high-volume, cost-sensitive tasks
- Use Claude for complex reasoning tasks that benefit from frontier capabilities
- NeMo's evaluation system measures both side-by-side

## 10. Limitations & Caveats

1. **No native integration**: Neither project explicitly supports the other.
   Integration relies on the shared MCP protocol.
2. **A2A gap**: Claude Agent SDK lacks native A2A support, requiring a wrapper.
3. **Auth model differences**: NeMo uses NVIDIA API keys; Claude uses Anthropic
   API keys. Both need to be configured when using both frameworks.
4. **Observability**: NeMo's LangSmith integration won't automatically trace
   Claude Agent SDK calls. You'd need separate observability for each.
