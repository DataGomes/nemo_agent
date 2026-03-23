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

### Pattern 5: Meta-Agent — Claude that optimizes Claude via NeMo
A Claude meta-agent that takes natural language descriptions ("I need a code
review agent, accuracy matters most, keep cost reasonable") and automatically
generates NeMo optimizer configs, eval datasets, runs optimization, interprets
the Pareto front, and deploys the optimized agent. No manual YAML needed.

### Pattern 6: Shared MCP tool ecosystem
Both frameworks can consume the same MCP tool servers, enabling a shared
tool layer across heterogeneous agent architectures.

## Quick Start

```bash
pip install -r requirements.txt
python test_compatibility.py
```

## Meta-Agent (Docker)

The meta-agent runs as a containerized FastAPI service with hot reload for development.

```bash
# 1. Set up your API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and NVIDIA_API_KEY

# 2. Start the container (dev mode with hot reload)
docker compose up

# 3. Describe what you want — the meta-agent does the rest
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"description": "I need a code review agent. Accuracy is critical, cost reasonable."}'

# Returns a job ID immediately:
# {"job_id": "a1b2c3d4", "status": "pending", ...}

# 4. Check job status / get results
curl http://localhost:8000/jobs/a1b2c3d4

# 5. Deploy to production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Hot Reload

In dev mode, the `examples/` directory is volume-mounted into the container.
Edit any file and uvicorn auto-restarts — no rebuild needed.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/optimize` | Start an optimization job |
| `GET` | `/jobs/{id}` | Get job status and results |
| `GET` | `/jobs` | List all jobs |
| `GET` | `/health` | Health check |

## Meta-Agent (CLI)

```bash
# Run the meta-agent interactively without Docker
pip install -r requirements.txt -r requirements-server.txt
python examples/meta_agent.py
```

## Files

- `Dockerfile` — Multi-stage build (builder + slim runtime)
- `docker-compose.yml` — Dev config (hot reload, volume mount)
- `docker-compose.prod.yml` — Production overlay (no reload, resource limits)
- `requirements.txt` — Core dependencies (NeMo + Claude SDK + MCP)
- `requirements-server.txt` — Server dependencies (FastAPI + uvicorn)
- `test_compatibility.py` — Import and API compatibility checks
- `examples/server.py` — FastAPI server wrapping the meta-agent
- `examples/meta_agent.py` — Meta-agent core (Claude optimizing Claude)
- `examples/nemo_mcp_to_claude.py` — Pattern 1: NeMo tools → Claude
- `examples/claude_tool_in_nemo.py` — Pattern 2: Claude inside NeMo
- `examples/nemo_ga_optimize_claude.py` — Pattern 4: GA prompt optimization
- `ANALYSIS.md` — Detailed compatibility analysis
