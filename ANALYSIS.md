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

## 8. Full Agent Optimization with GA + Optuna

### The Core Idea

NeMo's optimizer doesn't just optimize prompts — it can optimize the **entire
agent configuration**. The `OptimizableField` + `SearchSpace` system supports
three parameter types that together cover every "knob" on a Claude agent:

| Parameter Type | Optimizer | Examples |
|---------------|-----------|----------|
| **Prompts** (`is_prompt=True`) | Genetic Algorithm | System prompt, tool descriptions, few-shot examples |
| **Numerical** (`low`/`high`) | Optuna | Temperature, top_p, max_tokens, max_turns |
| **Categorical** (`values=[...]`) | Optuna | Model selection, tool sets, retrieval strategies |

### What You Can Optimize on a Claude Agent

```
┌──────────────────────────────────────────────────────────────┐
│                  NeMo Optimizer (nat optimize)                │
│                                                              │
│  GENETIC ALGORITHM (prompts):                                │
│    ├── system_prompt          "You are a code reviewer..."   │
│    ├── tool_descriptions      How tools are described        │
│    ├── few_shot_examples      Which examples to include      │
│    └── retrieval_query_tmpl   How to query the vector DB     │
│                                                              │
│  OPTUNA (numerical):                                         │
│    ├── temperature            0.0 → 1.0                      │
│    ├── top_p                  0.1 → 1.0                      │
│    ├── max_tokens             256 → 4096                     │
│    └── max_turns              1 → 10                         │
│                                                              │
│  OPTUNA (categorical):                                       │
│    ├── model_name             haiku / sonnet / opus          │
│    ├── tool_set               ["Read","Write"] / ["Bash"]    │
│    ├── retrieval_strategy     "semantic" / "hybrid" / "bm25" │
│    ├── chunk_size             256 / 512 / 1024               │
│    └── top_k_documents        3 / 5 / 10                     │
│                                                              │
│           ┌──────────────────────────┐                       │
│           │   Claude Agent SDK       │                       │
│           │   (runs each trial)      │                       │
│           └──────────────────────────┘                       │
│                      │                                       │
│                      ▼                                       │
│           Evaluators: correctness, latency, cost             │
│           Multi-objective Pareto optimization                │
└──────────────────────────────────────────────────────────────┘
```

### Model Selection (Haiku vs Sonnet vs Opus)

This is a **categorical `OptimizableField`**. Optuna treats it as a discrete
choice and explores which model gives the best score for your specific task:

```python
from nvidia_nat import OptimizableField, SearchSpace

class ClaudeAgentConfig(WorkflowConfig):
    model_name: str = OptimizableField(
        default="claude-sonnet-4-6",
        space=SearchSpace(values=[
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
        ]),
    )
    temperature: float = OptimizableField(
        default=1.0,
        space=SearchSpace(low=0.0, high=1.0),
    )
    tool_set: str = OptimizableField(
        default="full",
        space=SearchSpace(values=["minimal", "standard", "full"]),
    )
    system_prompt: str = OptimizableField(
        default="You are a helpful assistant.",
        space=SearchSpace(is_prompt=True),
    )
```

The optimizer might discover that **Haiku at temperature=0.3 with a minimal
tool set** outperforms **Opus at temperature=0.8 with full tools** for your
specific task — and costs 20x less.

### Tool Selection Optimization

Define tool sets as categorical values:

```python
TOOL_SETS = {
    "minimal": ["Read"],
    "standard": ["Read", "Write", "Bash"],
    "full": ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    "code_review": ["Read", "Grep", "Glob"],
}

class AgentConfig(WorkflowConfig):
    tool_set: str = OptimizableField(
        default="standard",
        space=SearchSpace(values=list(TOOL_SETS.keys())),
    )
```

The optimizer finds which tool set gives the best accuracy/cost tradeoff.
Fewer tools = fewer distractions for the model = potentially better results.

### Document Retrieval Optimization

For RAG-based agents, optimize retrieval parameters:

```python
class RAGConfig(WorkflowConfig):
    retrieval_strategy: str = OptimizableField(
        default="semantic",
        space=SearchSpace(values=["semantic", "hybrid", "bm25", "rerank"]),
    )
    top_k: int = OptimizableField(
        default=5,
        space=SearchSpace(low=1, high=20),
    )
    chunk_size: int = OptimizableField(
        default=512,
        space=SearchSpace(values=[256, 512, 1024, 2048]),
    )
    retrieval_prompt: str = OptimizableField(
        default="Find relevant context for: {query}",
        space=SearchSpace(is_prompt=True),
    )
```

### Full YAML Configuration Example

```yaml
optimizer:
  ga_generations: 10
  ga_population_size: 8
  n_trials_numeric: 30
  reps_per_param_set: 3

  parameters:
    # GA-optimized (prompts)
    system_prompt:
      type: prompt
      prompt_purpose: "Instruct a Claude agent to analyze code quality"
      initial_value: "You are a code review assistant..."

    tool_descriptions:
      type: prompt
      prompt_purpose: "Describe available tools to maximize correct tool selection"
      initial_value: "Read: Read file contents. Write: Create/overwrite files..."

    # Optuna-optimized (numerical)
    temperature:
      type: float
      low: 0.0
      high: 1.0

    top_p:
      type: float
      low: 0.1
      high: 1.0

    max_tokens:
      type: int
      low: 256
      high: 4096
      step: 256

    max_turns:
      type: int
      low: 1
      high: 10

    # Optuna-optimized (categorical)
    model_name:
      type: categorical
      values:
        - "claude-haiku-4-5-20251001"
        - "claude-sonnet-4-6"
        - "claude-opus-4-6"

    tool_set:
      type: categorical
      values: ["minimal", "standard", "full", "code_review"]

    retrieval_strategy:
      type: categorical
      values: ["semantic", "hybrid", "bm25"]

    top_k_documents:
      type: int
      low: 1
      high: 20

  eval_metrics:
    - name: correctness
      type: llm_judge
      weight: 0.5
    - name: latency
      type: builtin
      weight: 0.2
    - name: cost
      type: token_count
      weight: 0.3
```

### How the Optimizer Decides: User-Controlled Tradeoffs

The user fully controls **what matters** through eval metrics configuration.
NeMo provides two modes for handling competing objectives:

#### Mode 1: Weighted Single Score (Simple)

Assign a weight to each metric. The optimizer combines them into one composite
score and maximizes it. **You control what matters by adjusting the weights.**

```yaml
# Scenario A: "Accuracy is everything, I'll pay whatever it costs"
eval_metrics:
  - name: correctness
    type: llm_judge
    weight: 0.9
  - name: cost
    type: token_count
    weight: 0.05
  - name: latency
    type: builtin
    weight: 0.05
# → Optimizer will gravitate toward Opus + full tools + many retries
```

```yaml
# Scenario B: "Cost is king, accuracy just needs to be good enough"
eval_metrics:
  - name: correctness
    type: llm_judge
    weight: 0.3
  - name: cost
    type: token_count
    weight: 0.5
  - name: latency
    type: builtin
    weight: 0.2
# → Optimizer will find the cheapest config that still passes correctness
#   (probably Haiku + minimal tools + optimized prompt)
```

```yaml
# Scenario C: "Low latency for real-time UX, accuracy still important"
eval_metrics:
  - name: correctness
    type: llm_judge
    weight: 0.4
  - name: cost
    type: token_count
    weight: 0.1
  - name: latency
    type: builtin
    weight: 0.5
# → Optimizer will favor Haiku (fastest) with tuned prompts for accuracy
```

```yaml
# Scenario D: "Balance everything equally"
eval_metrics:
  - name: correctness
    type: llm_judge
    weight: 0.34
  - name: cost
    type: token_count
    weight: 0.33
  - name: latency
    type: builtin
    weight: 0.33
# → Optimizer finds the sweet spot — likely Sonnet with moderate settings
```

The composite score formula is:
`score = Σ(weight_i × normalized_metric_i)`

#### Mode 2: Multi-Objective Pareto Front (Advanced)

Instead of collapsing into one number, define multiple **independent objectives**.
Optuna uses multi-objective samplers (e.g., NSGA-II) to find the **Pareto
front** — the set of configs where improving one metric necessarily hurts another.

```yaml
optimizer:
  # Multi-objective: don't combine, find the tradeoff curve
  directions: ["maximize", "minimize", "minimize"]
  eval_metrics:
    - name: correctness   # maximize
      type: llm_judge
    - name: cost           # minimize
      type: token_count
    - name: latency        # minimize
      type: builtin
```

This produces a Pareto front like:

```
Accuracy ▲
  1.00 ── ● Config A: Opus, full tools, temp=0.1, 10 docs   ($0.08, 4.2s)
           │
  0.97 ── ● Config B: Opus, minimal tools, temp=0.2, 5 docs ($0.05, 3.1s)
           │
  0.95 ── ● Config C: Sonnet, standard tools, temp=0.2      ($0.012, 1.8s)
           │
  0.91 ── ● Config D: Sonnet, minimal tools, temp=0.0       ($0.008, 1.2s)
           │
  0.85 ── ● Config E: Haiku, minimal tools, temp=0.0        ($0.001, 0.4s)
           │
  0.70 ──   ✗ Config F: Haiku, full tools, temp=0.9         ($0.002, 0.6s)
           │   ^ DOMINATED: worse accuracy AND higher cost than E
           └──────────────────────────────────────────────────► Cost ($)
```

Configs A–E are **Pareto optimal** (can't improve one without hurting another).
Config F is **dominated** and gets eliminated. Then **you choose** based on
your business context:

- **Medical/legal app?** → Pick Config A (max accuracy, cost is secondary)
- **Consumer chatbot at scale?** → Pick Config D or E (cost-efficient)
- **Internal dev tool?** → Pick Config C (good balance)

#### Custom Evaluators

NeMo supports built-in and custom evaluators. For Claude agents, useful ones:

```yaml
eval_metrics:
  # Built-in evaluators
  - name: correctness
    type: llm_judge          # Uses an LLM to judge output quality
    judge_model: "claude-sonnet-4-6"  # Can use Claude as the judge too

  - name: latency
    type: builtin            # Measures wall-clock time

  - name: cost
    type: token_count        # Counts input + output tokens × model price

  # Custom evaluators (Python functions)
  - name: tool_efficiency
    type: custom
    evaluator_class: "my_evals.ToolEfficiencyEvaluator"
    # Scores based on: fewer tool calls = better

  - name: safety
    type: custom
    evaluator_class: "my_evals.SafetyEvaluator"
    # Checks for prompt injection resistance, PII leaks, etc.

  - name: format_compliance
    type: custom
    evaluator_class: "my_evals.FormatEvaluator"
    # Does the output match expected JSON/markdown/code format?
```

Example custom evaluator:

```python
from nvidia_nat import BaseEvaluator

class ToolEfficiencyEvaluator(BaseEvaluator):
    """Score based on how efficiently the agent uses tools."""

    def evaluate(self, workflow_output: dict) -> float:
        tool_calls = workflow_output.get("tool_call_count", 0)
        # Penalize excessive tool use (>5 calls gets score 0)
        return max(0.0, 1.0 - tool_calls / 5)

class CostEvaluator(BaseEvaluator):
    """Score based on actual API cost."""

    MODEL_COSTS_PER_1K = {
        "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
        "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
        "claude-opus-4-6": {"input": 0.015, "output": 0.075},
    }

    def evaluate(self, workflow_output: dict) -> float:
        model = workflow_output["model"]
        input_tokens = workflow_output["input_tokens"]
        output_tokens = workflow_output["output_tokens"]
        costs = self.MODEL_COSTS_PER_1K.get(model, {})
        total = (input_tokens / 1000 * costs.get("input", 0.003)
                 + output_tokens / 1000 * costs.get("output", 0.015))
        # Normalize: $0 = 1.0 score, $0.10+ = 0.0 score
        return max(0.0, 1.0 - total / 0.10)
```

#### Practical Decision Flowchart

```
Q: Do you have a single clear priority?
├── YES → Use weighted single score, give priority metric weight ≥ 0.5
│         Run: nat optimize --config workflow.yaml
│         Result: One best config
│
└── NO, multiple competing priorities →
    Q: Do you want to choose from tradeoffs?
    ├── YES → Use multi-objective Pareto
    │         Run: nat optimize --config workflow.yaml
    │         Result: Pareto front of configs to choose from
    │
    └── NO, just give me a good default →
        Use equal weights (0.33/0.33/0.33)
        Run: nat optimize --config workflow.yaml
        Result: Balanced best config
```

### Caveats for Full Agent Optimization

1. **Combinatorial explosion**: With model × tools × retrieval × prompts, the
   search space is large. Start with fewer parameters and expand incrementally.

2. **Cost**: Each trial = one Claude API call (× reps). A full sweep with 30
   Optuna trials + 10 GA generations × 8 population = ~110 trials × 3 reps =
   330 calls per eval question. Use Haiku for exploration, validate on target model.

3. **Rate limits**: Anthropic API rate limits may throttle large runs. Add
   retry/backoff in the workflow step. Run optimization off-peak.

4. **Caching**: NeMo's profiler tracks prompt-prefix overlap. Use this to
   design prompts with shared prefixes for Anthropic prompt caching (cost savings).

5. **Non-determinism**: Claude's outputs vary. The `reps_per_param_set` config
   averages over multiple runs to get statistically stable results.

6. **Model-specific behavior**: A prompt optimized for Haiku may not be optimal
   for Opus. Consider running separate optimization passes per model, or optimize
   model selection as a categorical param alongside prompt optimization.

7. **Eval quality matters most**: The optimizer is only as good as your evaluators.
   A bad correctness metric will lead to "optimized" configs that game the metric
   without actually being better. Invest in high-quality evaluation datasets and
   LLM judge prompts.

## 9. Meta-Agent: Claude That Optimizes Claude via NeMo

### The Idea

Instead of making the user manually configure optimizer YAML, eval datasets,
and metric weights — use a **Claude meta-agent** that does all of this
automatically. The user just describes what they want in plain English.

```
User: "I need a code review agent. Accuracy is critical, cost reasonable."
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  META-AGENT (Claude Agent SDK)                                       │
│                                                                      │
│  "The user wants accuracy-first code review. I'll set up:"           │
│                                                                      │
│  1. GENERATE CONFIG                                                  │
│     ├── models: [haiku, sonnet, opus]                                │
│     ├── tools: [minimal, standard, full, code_review_specific]       │
│     ├── temperature: 0.0 → 1.0                                      │
│     ├── eval weights: correctness=0.6, cost=0.25, latency=0.15      │
│     └── GA: evolve system prompts for code review domain             │
│                                                                      │
│  2. GENERATE EVAL DATASET                                            │
│     ├── 15 Python code snippets with known bugs                      │
│     ├── 10 TypeScript code snippets with known bugs                  │
│     ├── 5 clean code samples (should report "no issues")             │
│     └── 5 adversarial samples (tricky edge cases)                    │
│                                                                      │
│  3. RUN OPTIMIZATION (nat optimize)                                  │
│     └── NeMo runs GA + Optuna across full search space               │
│                                                                      │
│  4. INTERPRET RESULTS                                                │
│     "Sonnet at temp=0.1 with code_review tools scored 0.94 on        │
│      correctness at $0.02/review. Opus scored 0.96 but at $0.09.     │
│      Given your 'reasonable cost' constraint, I recommend Sonnet."   │
│                                                                      │
│  5. DEPLOY                                                           │
│     └── Write optimized_agent.py with final ClaudeAgentOptions       │
└──────────────────────────────────────────────────────────────────────┘
```

### How It Works

The meta-agent is itself a Claude Agent SDK agent with MCP tools that let it:
- **Write files** (optimizer configs, eval datasets)
- **Run NeMo** (`nat optimize`)
- **Read results** (parse Pareto front, best trials)
- **Deploy agents** (write final Python config)

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, tool, create_sdk_mcp_server

# The meta-agent's tools
@tool("generate_optimizer_config", "Write NeMo optimizer YAML", {...})
async def generate_optimizer_config(args): ...

@tool("generate_eval_dataset", "Create eval test cases", {...})
async def generate_eval_dataset(args): ...

@tool("run_nemo_optimizer", "Execute nat optimize", {...})
async def run_nemo_optimizer(args): ...

@tool("read_optimization_results", "Parse Pareto front", {...})
async def read_optimization_results(args): ...

@tool("deploy_optimized_agent", "Write final agent config", {...})
async def deploy_optimized_agent(args): ...

# The meta-agent itself
meta_server = create_sdk_mcp_server(
    name="meta-tools", version="1.0.0",
    tools=[generate_optimizer_config, generate_eval_dataset,
           run_nemo_optimizer, read_optimization_results,
           deploy_optimized_agent],
)

options = ClaudeAgentOptions(
    system_prompt="You are a meta-agent that optimizes other Claude agents...",
    mcp_servers={"meta": meta_server},
    allowed_tools=["mcp__meta__*"],
)

# User just says what they want
async with ClaudeSDKClient(options=options) as client:
    await client.query(
        "I need a code review agent. Accuracy is critical, cost reasonable."
    )
```

### What the Meta-Agent Infers from Natural Language

| User says | Meta-agent infers |
|-----------|-------------------|
| "accurate", "correct", "reliable" | correctness weight ≥ 0.6 |
| "cheap", "budget", "affordable" | cost weight ≥ 0.5 |
| "fast", "real-time", "instant" | latency weight ≥ 0.5 |
| "safe", "secure", "compliant" | adds safety evaluator |
| "code review" / "coding" | code_review tool set, code-focused eval dataset |
| "customer support" | FAQ-style eval dataset, format compliance evaluator |
| "research" / "summarize" | high max_tokens, retrieval optimization |
| "10K queries/day" | strong cost optimization, favor Haiku |
| No priorities stated | balanced weights (0.34/0.33/0.33) |

### Self-Optimizing Loop

The meta-agent can also **monitor and re-optimize** deployed agents:

```
┌─────────────────────────────────────────────┐
│  Deployed Claude Agent (optimized config)    │
│  - Serving production traffic                │
│  - Logging metrics to NeMo observability     │
└──────────────────┬──────────────────────────┘
                   │ metrics stream
                   ▼
┌─────────────────────────────────────────────┐
│  META-AGENT (scheduled check, e.g. daily)    │
│                                              │
│  "Correctness dropped from 0.94 to 0.87.    │
│   Looks like new code patterns in the eval   │
│   data that the current prompt doesn't       │
│   handle well.                               │
│                                              │
│   → Adding 10 new failure cases to dataset   │
│   → Re-running NeMo optimization             │
│   → New config: same model, updated prompt   │
│   → Hot-swapping agent config"               │
└─────────────────────────────────────────────┘
```

This creates a **closed-loop optimization system**: Claude agents that
continuously improve themselves via NeMo, managed by another Claude agent.

See `examples/meta_agent.py` for the full implementation.

## 10. GRPO / Reinforcement Learning (Advanced)

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
