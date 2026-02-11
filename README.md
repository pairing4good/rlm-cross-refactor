
---

<h1 align="center" style="font-size:2.8em">
<span>Recursive Language Models (<span style="color:orange">RLM</span>s)</span>
</h1>

<p align="center" style="font-size:1.3em">
  <a href="https://arxiv.org/abs/2512.24601">Full Paper</a> •
  <a href="https://alexzhang13.github.io/blog/2025/rlm/">Blogpost</a> •
  <a href="https://alexzhang13.github.io/rlm/">Documentation</a> •
  <a href="https://github.com/alexzhang13/rlm-minimal">RLM Minimal</a>
</p>

<p align="center">
  <a href="https://github.com/alexzhang13/rlm/actions/workflows/style.yml">
    <img src="https://github.com/alexzhang13/rlm/actions/workflows/style.yml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/alexzhang13/rlm/actions/workflows/test.yml">
    <img src="https://github.com/alexzhang13/rlm/actions/workflows/test.yml/badge.svg" alt="Test" />
  </a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.24601">
    <img src="media/paper_preview.png" alt="Paper Preview" width="300"/>
  </a>
</p>

## Overview
Recursive Language Models (RLMs) are a task-agnostic inference paradigm for language models (LMs) to handle near-infinite length contexts by enabling the LM to *programmatically* examine, decompose, and recursively call itself over its input. RLMs replace the canonical `llm.completion(prompt, model)` call with a `rlm.completion(prompt, model)` call. RLMs offload the context as a variable in a REPL environment that the LM can interact with and launch sub-LM calls inside of.

This repository provides an extensible inference engine for using RLMs around standard API-based and local LLMs. The initial experiments and idea were proposed in a [blogpost](https://alexzhang13.github.io/blog/2025/rlm/) in 2025, with expanded results in an [arXiv preprint](https://arxiv.org/abs/2512.24601).

> [!NOTE]
> This repository contains inference code for RLMs with support for various sandbox environments. Open-source contributions are welcome. This repository is maintained by the authors of the paper from the MIT OASYS lab.

## Quick Setup
You can try out RLMs quickly by installing from PyPi:
```bash
pip install rlms
```

> [!IMPORTANT]
> **Token Limit & Cost Protection**: RLM has **default limits of 1,000,000 tokens** for both root agent and sub-agent calls to prevent unexpectedly large bills. This costs approximately:
> - GPT-3.5 Turbo: ~$1 per session
> - GPT-4o: ~$6 per session  
> - GPT-4 Turbo: ~$20 per session
> - Claude Opus: ~$45 per session
>
> **Configure limits via RLM constructor parameters** (NOT in `backend_kwargs`):
> ```python
> # ✅ Correct
> rlm = RLM(
>     backend_kwargs={"model_name": "gpt-4o"},  # No token limits here
>     max_root_tokens=500_000,  # Session limit for root agent
>     max_sub_tokens=500_000,   # Session limit for sub-agents
> )
> 
> # ❌ Wrong - will cause validation error
> rlm = RLM(
>     backend_kwargs={
>         "model_name": "gpt-4o",
>         "max_root_tokens": 500_000,  # Don't do this!
>     }
> )
> ```
> 
> **Root vs Sub-Agent Tokens**: 
> - `max_root_tokens`: Limits tokens used by the main orchestration agent (depth=0)
> - `max_sub_tokens`: Limits cumulative tokens used by `llm_query()` calls in generated code (depth=1)
>
> **Minimum Token Requirements**: Each limit must be at least **50 tokens** (typical minimum for a single LM call). 
> Limits below this threshold will cause immediate termination before any calls are made.

The default RLM client uses a REPL environment that runs on the host process through Python `exec` calls. It uses the same virtual environment as the host process (i.e. it will have access to the same dependencies), but with some limitations in its available global modules. As an example, we can call RLM completions using GPT-5-nano:
```python
from rlm import RLM

rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-nano"},
    verbose=True,  # For printing to console with rich, disabled by default.
)

print(rlm.completion("Print me the first 100 powers of two, each on a newline.").response)
```

**Understanding Your Token Budget:**
- Each session (one `completion()` call) has separate limits for root and sub-agents (1M tokens each by default)
- You can check token usage: `result.usage_summary.to_dict()`
- Adjust limits with `max_root_tokens` and `max_sub_tokens` parameters (see [Token Limits](#important-token-limit--cost-protection) above)

<details>
<summary><b>Manual Setup</b></summary>

Set up the dependencies with `uv` (or your virtual environment of choice):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init && uv venv --python 3.12  # change version as needed
uv pip install -e .
```

This project includes a `Makefile` to simplify common tasks.

- `make install`: Install base dependencies.
- `make check`: Run linter, formatter, and tests.

To run a quick test, the following will run an RLM query with the OpenAI client using your environment variable `OPENAI_API_KEY` (feel free to change this). This will generate console output as well as a log which you can use with the visualizer to explore the trajectories.
```bash
make quickstart
```

</details>

## REPL Environments
We support two types of REPL environments -- isolated, and non-isolated. Non-isolated environments (default) run code execution on the same machine as the RLM (e.g. through `exec`), which is pretty reasonable for some local low-risk tasks, like simple benchmarking, but can be problematic if the prompts or tool calls can interact with malicious users. Fully isolated environments used Cloud-based sandboxes (e.g. Prime Sandboxes, [Modal Sandboxes](https://modal.com/docs/guide/sandboxes)) to run code generated by the RLM, ensuring completely isolation from the host process. Environments can be added, but we natively support the following: `local` (default), `modal`, `prime`.

```python
rlm = RLM(
    environment="...", # "local", "docker", "modal", "prime"
    environment_kwargs={...},
)
```

### Local Environments
The default `local` environment `LocalREPL` runs in the same process as the RLM itself, with specified global and local namespaces for minimal security. Using this REPL is generally safe, but should not be used for production settings. It also shares the same virtual environment (e.g. Conda or uv) as the host process.

#### Docker <img src="https://github.com/docker.png" alt="Docker" height="20" style="vertical-align: middle;"/> (*requires [Docker installed](https://docs.docker.com/desktop/setup/install/)*)
We also support a Docker-based environment called `DockerREPL` that launches the REPL environment as a Docker image. By default, we use the `python:3.11-slim` image, but the user can specify custom images as well.

### Isolated Environments
We support several different REPL environments that run on separate, cloud-based machines. Whenever a recursive sub-call is made in these instances, it is requested from the host process.

#### Modal Sandboxes <img src="https://github.com/modal-labs.png" alt="Modal" height="20" style="vertical-align: middle;"/>
To use [Modal Sandboxes](https://modal.com/docs/guide/sandboxes) as the REPL environment, you need to install and authenticate your Modal account.
```bash
uv add modal  # add modal library
modal setup   # authenticate account
```

#### Prime Intellect Sandboxes <img src="https://github.com/PrimeIntellect-ai.png" alt="Prime Intellect" height="20" style="vertical-align: middle;"/>
> [!NOTE]
> **Prime Intellect Sandboxes** are currently a beta feature. See the [documentation](https://docs.primeintellect.ai/sandboxes/overview) for more information. We noticed slow runtimes when using these sandboxes, which is currently an open issue.


To use [Prime Sandboxes](https://docs.primeintellect.ai/sandboxes/sdk), install the SDK and set your API key:
```bash
uv pip install -e ".[prime]"
export PRIME_API_KEY=...
```


### Model Providers
We currently support most major clients (OpenAI, Anthropic, AWS Bedrock), as well as the router platforms (OpenRouter, Portkey, LiteLLM). For local models, we recommend using vLLM (which interfaces with the [OpenAI client](https://github.com/alexzhang13/rlm/blob/main/rlm/clients/openai.py)). To view or add support for more clients, start by looking at [`rlm/clients/`](https://github.com/alexzhang13/rlm/tree/main/rlm/clients).

#### AWS Bedrock
To use AWS Bedrock models (Claude, Llama, Mistral, Titan), ensure you have AWS credentials configured via IAM role, AWS CLI, or environment variables:
```bash
uv pip install boto3
```

```python
from rlm import RLM

rlm = RLM(
    backend="bedrock",
    backend_kwargs={
        "model_name": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "region_name": "us-east-1",
        # Optional: aws_access_key_id, aws_secret_access_key
    },
    environment="local"
)
```

## Multi-Tier Model Routing for Cost Optimization

RLM supports **multi-tier routing** where you can configure a powerful root model for high-level reasoning and delegate simpler execution tasks to more cost-effective sub-models. This can reduce costs by **10-20x** for tasks involving many sub-queries.

### How It Works

When the root model generates code containing `llm_query()` calls, those sub-queries are automatically routed to your configured sub-model (at depth=1):

```python
from rlm import RLM

rlm = RLM(
    # Root model (depth=0): Handles main reasoning and planning
    backend="anthropic",
    backend_kwargs={"model_name": "claude-3-5-sonnet-20241022"},
    
    # Sub-model (depth=1): Handles llm_query() calls
    other_backends=["anthropic"],
    other_backend_kwargs=[{"model_name": "claude-3-haiku-20240307"}],
)

# The root model (Sonnet) will plan the approach
# Any llm_query() calls will automatically use the sub-model (Haiku)
result = rlm.completion("""
Analyze this dataset by processing each item:
1. Extract key fields from each record
2. Summarize findings
""")
```

### Cost Comparison: Sonnet vs Haiku

| Model | Input Cost | Output Cost | Use Case |
|-------|-----------|------------|----------|
| Claude 3.5 Sonnet | $3/M tokens | $15/M tokens | Complex reasoning, planning |
| Claude 3 Haiku | $0.25/M tokens | $1.25/M tokens | Simple extraction, formatting |

**Savings Example:**
- Root model makes 1 planning call (Sonnet): ~10K tokens = $0.15
- Sub-model makes 100 extraction calls (Haiku): ~500K tokens = $0.75
- **Total: $0.90** vs **$9.00 all-Sonnet** (90% savings!)

### Recommended Model Pairs by Provider

| Provider | Root Model | Sub-Model | Cost Ratio |
|----------|-----------|-----------|-----------|
| **Anthropic** | `claude-3-5-sonnet-20241022` | `claude-3-haiku-20240307` | 10x cheaper |
| **OpenAI** | `gpt-4o` | `gpt-4o-mini` | 20x cheaper |
| **Bedrock** | `us.anthropic.claude-sonnet-4-*` | `anthropic.claude-3-5-haiku-*` | 10x cheaper + volume discounts |
| **Gemini** | `gemini-2.0-pro` | `gemini-2.5-flash` | 5-10x cheaper |
| **Azure OpenAI** | `gpt-4o` | `gpt-4o-mini` | 20x cheaper |

### When to Use Multi-Tier Routing

✅ **Good use cases:**
- Processing large datasets with many sub-queries
- Extract-transform-load (ETL) operations
- Batch processing with simple per-item operations
- Analysis requiring many lookups/validations

❌ **Not needed for:**
- Single-shot completions without sub-queries
- Tasks where every step requires advanced reasoning
- Small-scale processing (<10 sub-queries)

See [examples/multi_tier_anthropic.py](examples/multi_tier_anthropic.py) for a complete working example.

## Relevant Reading
* **[Dec '25]** [Recursive Language Models arXiv](https://arxiv.org/abs/2512.24601)
* **[Oct '25]** [Recursive Language Models Blogpost](https://alexzhang13.github.io/blog/2025/rlm/)

If you use this code or repository in your research, please cite:

```bibtex
@misc{zhang2025recursivelanguagemodels,
      title={Recursive Language Models}, 
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2025},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601}, 
}
```

## Optional Debugging: Visualizing RLM Trajectories
We additionally provide a simple visualizer tool to examine and view the code, sub-LM, and root-LM calls of an RLM trajectory. To save log files (`.jsonl`) on every completion call that can be viewed in the visualizer, initialize the `RLMLogger` object and pass it into the `RLM` on initialization:
```python
from rlm.logger import RLMLogger
from rlm import RLM

logger = RLMLogger(log_dir="./logs")
rlm = RLM(
    ...
    logger=logger
)
```

To run the visualizer locally, we use Node.js and shadcn/ui:
```
cd visualizer/
npm run dev        # default localhost:3001
```

You'll have the option to select saved `.jsonl` files 
<p align="center">
  <img src="media/visualizer.png" alt="RLM Visualizer Example" width="800"/>
</p>
