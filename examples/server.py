"""
Meta-Agent API Server

A FastAPI service that wraps the meta-agent. Users POST a natural language
description of the agent they want, and the meta-agent handles the full
NeMo optimization pipeline.

Endpoints:
    POST /optimize        — Start an optimization job
    GET  /jobs/{job_id}   — Check job status / get results
    GET  /jobs            — List all jobs
    GET  /health          — Health check
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from meta_agent import (
    META_AGENT_SYSTEM_PROMPT,
    generate_optimizer_config,
    generate_eval_dataset,
    run_nemo_optimizer,
    read_optimization_results,
    deploy_optimized_agent,
)

# Lazy imports — these may not be installed in all environments
try:
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        create_sdk_mcp_server,
    )
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


app = FastAPI(
    title="Meta-Agent: Claude Optimizing Claude via NeMo",
    version="0.1.0",
    description="Describe the agent you want. The meta-agent builds and optimizes it.",
)

WORK_DIR = Path("/app/workdir")
WORK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Models
# ============================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizeRequest(BaseModel):
    description: str = Field(
        ...,
        description="Natural language description of the agent you want",
        examples=[
            "I need a code review agent. Accuracy is critical, cost reasonable.",
            "Build a customer support bot. Fast and cheap, 10K queries/day.",
        ],
    )
    # Optional overrides for power users
    models: Optional[list[str]] = Field(
        None,
        description="Override model search space (default: all Claude tiers)",
    )
    max_eval_examples: Optional[int] = Field(
        None,
        description="Max eval examples to generate (default: 30)",
    )


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    description: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    logs: list[str] = []


# In-memory job store (swap for Redis/DB in production)
jobs: dict[str, JobInfo] = {}


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sdk_available": HAS_SDK,
        "work_dir": str(WORK_DIR),
    }


@app.post("/optimize", response_model=JobInfo)
async def start_optimization(request: OptimizeRequest):
    """Start an optimization job. Returns immediately with a job ID."""
    job_id = str(uuid.uuid4())[:8]
    job = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        description=request.description,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    jobs[job_id] = job

    # Run optimization in background
    asyncio.create_task(_run_optimization(job_id, request))

    return job


@app.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: str):
    """Get the status and results of an optimization job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return jobs[job_id]


@app.get("/jobs", response_model=list[JobInfo])
async def list_jobs():
    """List all optimization jobs."""
    return list(jobs.values())


# ============================================================================
# Background optimization task
# ============================================================================

async def _run_optimization(job_id: str, request: OptimizeRequest):
    """Execute the full meta-agent optimization pipeline."""
    job = jobs[job_id]
    job.status = JobStatus.RUNNING
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        job.logs.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}")

    try:
        if not HAS_SDK:
            # Fallback: generate config without running the full agent loop
            log("Claude Agent SDK not available — running in config-generation-only mode")
            await _run_config_only_mode(job_id, request, job_dir, log)
            return

        log("Starting meta-agent")

        meta_server = create_sdk_mcp_server(
            name="meta-tools",
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
            mcp_servers={"meta": meta_server},
            allowed_tools=[
                "mcp__meta__generate_optimizer_config",
                "mcp__meta__generate_eval_dataset",
                "mcp__meta__run_nemo_optimizer",
                "mcp__meta__read_optimization_results",
                "mcp__meta__deploy_optimized_agent",
            ],
            max_turns=15,
        )

        prompt = (
            f'The user wants you to create and optimize an AI agent.\n'
            f'Request: "{request.description}"\n'
            f'Work directory: {job_dir}\n'
            f'Models override: {request.models or "auto"}\n'
            f'Max eval examples: {request.max_eval_examples or 30}\n\n'
            f'Run the full pipeline: config → eval dataset → optimize → '
            f'interpret → deploy.'
        )

        log("Meta-agent processing request...")
        agent_output = []

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if hasattr(message, "content"):
                    for block in message.content:
                        if hasattr(block, "text"):
                            agent_output.append(block.text)
                            log(f"Agent: {block.text[:200]}")

        # Collect results
        results = {}
        config_path = job_dir / "optimizer.yaml"
        if config_path.exists():
            results["config"] = config_path.read_text()
        dataset_path = job_dir / "eval_dataset.json"
        if dataset_path.exists():
            results["eval_dataset"] = json.loads(dataset_path.read_text())
        agent_path = job_dir / "optimized_agent.py"
        if agent_path.exists():
            results["optimized_agent"] = agent_path.read_text()
        results["agent_output"] = "\n".join(agent_output)

        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc).isoformat()
        job.result = results
        log("Optimization complete")

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now(timezone.utc).isoformat()
        log(f"Failed: {e}")


async def _run_config_only_mode(
    job_id: str,
    request: OptimizeRequest,
    job_dir: Path,
    log,
):
    """Fallback mode: generate config files without running the full agent."""
    job = jobs[job_id]
    log("Generating optimizer config template based on request...")

    # Simple heuristic config generation
    description = request.description.lower()

    # Infer weights from description
    correctness_w = 0.34
    cost_w = 0.33
    latency_w = 0.33

    if any(w in description for w in ["accurate", "correct", "reliable", "critical"]):
        correctness_w = 0.6
        cost_w = 0.25
        latency_w = 0.15
    elif any(w in description for w in ["cheap", "budget", "affordable", "cost"]):
        cost_w = 0.5
        correctness_w = 0.35
        latency_w = 0.15
    elif any(w in description for w in ["fast", "real-time", "instant", "latency"]):
        latency_w = 0.5
        correctness_w = 0.35
        cost_w = 0.15

    models = request.models or ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"]

    config = {
        "search_space": {
            "model": {"type": "categorical", "values": models},
            "temperature": {"type": "float", "low": 0.0, "high": 1.0},
            "max_turns": {"type": "int", "low": 1, "high": 10},
        },
        "eval_metrics": {
            "correctness": {"weight": correctness_w},
            "cost": {"weight": cost_w},
            "latency": {"weight": latency_w},
        },
        "optimization": {
            "n_trials": 30,
            "ga_generations": 10,
            "ga_population": 8,
            "reps_per_param_set": 3,
        },
        "original_request": request.description,
    }

    config_path = job_dir / "optimizer_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    job.status = JobStatus.COMPLETED
    job.completed_at = datetime.now(timezone.utc).isoformat()
    job.result = {
        "config": config,
        "message": (
            "Generated optimizer config. Install claude-agent-sdk and nvidia-nat "
            "to run the full meta-agent pipeline. Or run: "
            "nat optimize --config_file " + str(config_path)
        ),
    }
    log("Config generated (SDK not available for full pipeline)")
