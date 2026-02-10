# RLM Architecture Documentation

## Table of Contents
- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [System Architecture](#system-architecture)
- [Component Details](#component-details)
- [Communication Patterns](#communication-patterns)
- [Data Flow](#data-flow)
- [Environment Types](#environment-types)
- [Extension Points](#extension-points)

---

## Overview

**Recursive Language Models (RLMs)** is an inference paradigm that enables language models to handle extremely large contexts by allowing them to programmatically decompose problems and recursively call themselves. Instead of stuffing an entire massive context into a single prompt, RLMs provide the LM with:

1. **A REPL environment** where the context lives as a variable
2. **The ability to execute Python code** to examine and manipulate that context
3. **Recursive LM query functions** (`llm_query`, `llm_query_batched`) to launch sub-LM calls with processed chunks

This transforms the standard `llm.completion(prompt, model)` call into an iterative, programmatic problem-solving loop.

### Key Innovation
RLMs solve the context length problem by offloading context examination and decomposition to code execution, allowing the LM to:
- Inspect context programmatically (slice, filter, search)
- Break large contexts into intelligent chunks
- Launch parallel sub-LM queries on chunks
- Aggregate results through iterative processing
- Maintain state across iterations via REPL variables

---

## Core Concepts

### 1. RLM (Recursive Language Model)
The main entry point that orchestrates the entire system. Each `rlm.completion()` call:
- Spawns an LM Handler (socket server for LM queries)
- Creates/reuses a REPL environment
- Enters an iterative loop where the LM generates responses containing:
  - Reasoning and planning
  - Python code blocks (```repl```)
  - Sub-LM query calls
  - Final answers when complete

### 2. REPL Environment
An execution environment (local, Docker, or cloud sandbox) where:
- User's context is loaded as a variable (`context`, `context_0`, etc.)
- Generated Python code executes
- Special functions are available:
  - `llm_query(prompt, model=None)` - Make single sub-LM calls
  - `llm_query_batched(prompts, model=None)` - Parallel sub-LM calls
  - `FINAL_VAR(variable_name)` - Return a variable as final answer
  - `SHOW_VARS()` - List available variables
- State persists across iterations within a completion
- (Optional) Multi-turn persistence for conversations

### 3. LM Handler
A multi-threaded TCP socket server that:
- Receives LM query requests from REPL environments
- Routes requests to appropriate LM clients (OpenAI, Anthropic, etc.)
- Handles both single and batched requests concurrently
- Tracks usage metrics (tokens, costs)
- Returns responses to the REPL environment

### 4. Iteration Loop
Each completion runs up to `max_iterations` (default: 30) cycles:
```
1. Prompt LM with system prompt + message history + context metadata
2. LM responds with reasoning + code blocks
3. Execute code blocks in REPL, capture outputs
4. Check for FINAL(...) or FINAL_VAR(...) answer
5. If final answer found → return, else append execution results to history and repeat
```

### 5. Depth
Controls recursive nesting (currently supports depth 0 and 1):
- **Depth 0**: Root RLM can make sub-calls (depth 1)
- **Depth 1**: Sub-calls cannot recurse further (act as regular LMs)
- At `max_depth`, RLM falls back to standard LM completion

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Application                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  rlm = RLM(backend="openai", environment="local")                 │  │
│  │  result = rlm.completion("Analyze this huge document...")         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          RLM Core (rlm.py)                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  • Iteration loop (max_iterations)                               │   │
│  │  • Message history management                                    │   │
│  │  • Final answer detection                                        │   │
│  │  • Logging & verbose output                                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└───────────┬─────────────────────────────────┬───────────────────────────┘
            │                                 │
            ▼                                 ▼
┌────────────────────────────┐   ┌───────────────────────────────────────┐
│      LM Handler            │   │      REPL Environment                 │
│   (lm_handler.py)          │   │      (base_env.py)                    │
│  ┌──────────────────────┐  │   │  ┌─────────────────────────────────┐ │
│  │ ThreadingTCPServer   │  │   │  │ • Persistent namespace          │ │
│  │ (127.0.0.1:PORT)     │◄─┼───┼──┤ • Context loading               │ │
│  │                      │  │   │  │ • Code execution (exec)         │ │
│  │ Routes to LM Clients │  │   │  │ • llm_query() → socket request  │ │
│  └──────────────────────┘  │   │  └─────────────────────────────────┘ │
│            │               │   │                                       │
│            ▼               │   │  Types:                               │
│  ┌──────────────────────┐  │   │  • LocalREPL (local process)        │
│  │   BaseLM Clients:    │  │   │  • DockerREPL (container)           │
│  │   • OpenAI           │  │   │  • ModalREPL (Modal sandboxes)      │
│  │   • Anthropic        │  │   │  • PrimeREPL (Prime sandboxes)      │
│  │   • Gemini           │  │   │  • DaytonaREPL (Daytona envs)       │
│  │   • LiteLLM          │  │   │                                       │
│  │   • Portkey          │  │   │                                       │
│  └──────────────────────┘  │   │                                       │
└────────────────────────────┘   └───────────────────────────────────────┘
```

### Component Hierarchy

```
RLM
├── LMHandler (TCP Server)
│   ├── BaseLM Clients
│   │   ├── OpenAI
│   │   ├── Anthropic
│   │   ├── Gemini
│   │   └── ...
│   └── Usage Tracking
│
├── Environment (REPL)
│   ├── NonIsolatedEnv
│   │   ├── LocalREPL
│   │   └── DockerREPL
│   └── IsolatedEnv
│       ├── ModalREPL
│       ├── PrimeREPL
│       └── DaytonaREPL
│
├── Message History Management
├── Iteration Loop
└── Logger (optional)
```

---

## Component Details

### RLM (`rlm/core/rlm.py`)

**Responsibilities:**
- Orchestrate the complete inference lifecycle
- Manage iteration loop and message history
- Spawn/cleanup LM Handler and Environment per completion
- Parse responses for code blocks and final answers
- Handle persistent multi-turn sessions (optional)

**Key Methods:**
- `completion(prompt)` - Main entry point for RLM queries
- `_spawn_completion_context()` - Context manager that sets up handler + environment
- `_completion_turn()` - Execute one iteration (prompt → response → code execution)
- `_setup_prompt()` - Build system prompt with metadata
- `_default_answer()` - Fallback if max iterations reached
- `_fallback_answer()` - Handle depth limit (become regular LM)

**Configuration:**
```python
RLM(
    backend="openai",                    # LM client to use (root model at depth=0)
    backend_kwargs={"model_name": "..."},
    environment="local",                 # REPL type
    environment_kwargs={},
    depth=0,                            # Current depth (0=root)
    max_depth=1,                        # Recursion limit
    max_iterations=30,                  # Iteration limit
    custom_system_prompt=None,          # Override default prompt
    other_backends=["openai"],          # Sub-model clients for depth-based routing
    other_backend_kwargs=[{...}],       # Config for sub-models (currently supports 1)
    logger=None,                        # RLMLogger instance
    verbose=False,                      # Pretty console output
    persistent=False                    # Multi-turn conversations
)
```

**Multi-Tier Routing (`other_backends`):**

The `other_backends` parameter enables cost optimization by routing sub-queries (made via `llm_query()`) to different, typically cheaper models:

- **Root model (depth=0)**: Uses `backend` + `backend_kwargs` for main reasoning
- **Sub-model (depth=1)**: Uses `other_backends[0]` + `other_backend_kwargs[0]` for `llm_query()` calls

**Example - Cost-Optimized Configuration:**
```python
rlm = RLM(
    # Powerful model for strategic planning
    backend="anthropic",
    backend_kwargs={"model_name": "claude-3-5-sonnet-20241022"},
    
    # Cost-effective model for execution tasks
    other_backends=["anthropic"],
    other_backend_kwargs=[{"model_name": "claude-3-haiku-20240307"}],
)
# Sonnet plans, Haiku executes → 10x cost reduction on sub-queries!
```

**When sub-model is used:**
- When generated code calls `llm_query(prompt)` at depth=0
- These calls are routed to depth=1 (the sub-model)
- Useful for: batch processing, extraction, validation, simple transformations

**Note:** Currently limited to 1 sub-model. Future versions will support multiple tiers.

---

### LM Handler (`rlm/core/lm_handler.py`)

**Architecture:**
```
LMHandler
├── ThreadingTCPServer (multi-threaded socket server)
│   └── LMRequestHandler (per-request handler)
├── Client Registry (dict[model_name -> BaseLM])
├── Default Client (for unspecified model requests)
└── Other Backend Client (for depth-based routing)
```

**Responsibilities:**
- Run TCP server on `127.0.0.1:{auto_assigned_port}`
- Accept JSON requests with length-prefix protocol
- Route requests to appropriate LM clients
- Handle both single and batched requests concurrently
- Track usage metrics across all calls

**Protocol Format:**
```
4-byte big-endian length prefix + UTF-8 JSON payload
```

**Request/Response Types:**
```python
# Request (sent by REPL environment)
{
    "prompt": str | dict,              # Single request
    "prompts": list[str | dict],       # Batched request
    "model": str | None,               # Optional model override
    "depth": int                       # Current depth
}

# Response (from LM Handler)
{
    "chat_completion": {...},          # Single response
    "chat_completions": [{...}],       # Batched response
    "error": str | None                # Error message
}
```

**Client Selection Logic (Multi-Tier Routing):**

The handler routes requests to different LM clients based on explicit model name or depth:

1. **Explicit model name** → use registered client for that model (highest priority)
2. **Depth-based routing** → if `depth == 1` and `other_backend_client` exists → use sub-model
3. **Fallback** → use default (root) client

**Example Routing Flow:**
```
┌─────────────────────────────────────────────────────────┐
│ Root RLM (depth=0)                                      │
│ Model: Claude Sonnet ($3/M in, $15/M out)              │
│                                                         │
│ Generates code:                                         │
│   results = []                                          │
│   for item in data:                                     │
│       # This llm_query() triggers depth=1 routing       │
│       summary = llm_query(f"Summarize: {item}")         │
│       results.append(summary)                           │
└────────────────────┬────────────────────────────────────┘
                     │ llm_query() at depth=0
                     ▼
┌─────────────────────────────────────────────────────────┐
│ LM Handler                                              │
│ Receives: {"prompt": "Summarize: ...", "depth": 1}     │
│                                                         │
│ Routing decision:                                       │
│   • depth=1 detected                                    │
│   • other_backend_client exists                         │
│   → Route to sub-model (Claude Haiku)                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Sub-Model (depth=1)                                     │
│ Model: Claude Haiku ($0.25/M in, $1.25/M out)          │
│                                                         │
│ Processes simple extraction/summarization               │
│ Returns result to root RLM                              │
│                                                         │
│ Cost: ~90% cheaper for sub-queries!                     │
└─────────────────────────────────────────────────────────┘
```

**Cost Optimization Strategy:**
- Root model: Use powerful model (Sonnet, GPT-4o) for planning/reasoning
- Sub-model: Use cost-effective model (Haiku, GPT-4o-mini) for execution
- Savings: 10-20x reduction on sub-query costs

**Key Methods:**
- `start()` - Spawn server thread
- `stop()` - Shutdown server
- `register_client(model_name, client)` - Add client to registry
- `get_client(model, depth)` - Route requests to appropriate client
- `get_client(model, depth)` - Resolve which client to use
- `completion(prompt)` - Direct completion call (used by RLM for root queries)
- `get_usage_summary()` - Aggregate usage across all clients

---

### Environments (`rlm/environments/`)

#### Base Classes

**BaseEnv** (abstract)
- `setup()` - Initialize globals, locals, helper functions
- `load_context(context_payload)` - Make context available to code
- `execute_code(code: str) -> REPLResult` - Run code, return stdout/stderr/error

**NonIsolatedEnv** (same machine as RLM)
- Easier setup, faster execution
- Less isolation (security concerns for untrusted code)
- Examples: LocalREPL, DockerREPL

**IsolatedEnv** (cloud sandboxes)
- Complete isolation, own machine
- Slower due to network overhead
- Examples: ModalREPL, PrimeREPL, DaytonaREPL

#### LocalREPL (`local_repl.py`)

The default, simplest environment. Runs code via Python `exec()` in a sandboxed namespace.

**Key Features:**
- Sandboxed globals with `_SAFE_BUILTINS` (blocks `eval`, `exec`, `input`, etc.)
- Temporary working directory (auto-cleaned)
- Persistent namespace across code blocks within a completion
- Direct socket communication with LM Handler
- Supports multi-turn persistence (implements `SupportsPersistence`)

**Namespace Setup:**
```python
self.globals = {
    "__builtins__": _SAFE_BUILTINS,
    "__name__": "__main__",
    "FINAL_VAR": self._final_var,
    "SHOW_VARS": self._show_vars,
    "llm_query": self._llm_query,
    "llm_query_batched": self._llm_query_batched,
    "context": <user_payload>,          # First context
    "context_0": <user_payload>,        # Versioned contexts
    # ... additional contexts in multi-turn
}
self.locals = {}  # Variables created by executed code
```

**Code Execution Flow:**
```python
def execute_code(self, code: str) -> REPLResult:
    # 1. Capture stdout/stderr
    # 2. Change to temp directory
    # 3. Clear pending LLM calls list
    # 4. Execute: exec(code, self.globals, self.locals)
    # 5. Collect any LLM calls made during execution
    # 6. Return REPLResult(stdout, stderr, error, llm_calls)
```

**Multi-Turn Persistence** (implements `SupportsPersistence`):
- `update_handler_address(address)` - Update for new completion context
- `add_context(payload, index=None)` - Add `context_N` variable (auto-increment or specify index)
- `get_context_count()` - Return number of contexts loaded
- `add_history(messages, index=None)` - Store previous message histories as `history_N`
- `get_history_count()` - Return number of histories stored

**Versioning:**
```python
# First completion
context = context_0 = "first prompt"

# Second completion (persistent=True)
context = context_0 = "first prompt"    # Alias always points to index 0
context_1 = "second prompt"
history_0 = [...]                       # Message history from first completion
```

#### ModalREPL (`modal_repl.py`)

Uses Modal Sandboxes for complete isolation on Modal's infrastructure.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│  Host Process (ModalREPL)                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Poller Thread (background)                             │    │
│  │  • Poll {tunnel_url}/pending every 100ms                │    │
│  │  • Forward requests to LMHandler via socket             │    │
│  │  • POST responses to {tunnel_url}/respond               │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP (encrypted tunnel)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Modal Sandbox                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Broker Server (Flask, port 8080)                       │    │
│  │  Endpoints:                                             │    │
│  │    POST /enqueue   - sandbox code submits LLM request   │    │
│  │    GET  /pending   - host polls for pending requests    │    │
│  │    POST /respond   - host submits responses             │    │
│  │    GET  /health    - health check                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Execution Script (per code block)                      │    │
│  │  • Load state from /tmp/rlm_state.dill                  │    │
│  │  • Execute code                                         │    │
│  │  • llm_query() → POST to localhost:8080/enqueue        │    │
│  │  • Save state to /tmp/rlm_state.dill                    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Broker Server** (Flask app inside sandbox)
   - Queue for pending LLM requests
   - Blocking `/enqueue` endpoint (waits for response via `threading.Event`)
   - Non-blocking `/pending` and `/respond` for host polling

2. **Poller Thread** (runs on host)
   - Background daemon polling `/pending` every 100ms
   - Forwards requests to LMHandler via socket
   - Returns responses via `/respond`

3. **Execution Script** (runs per code block)
   - Serializes/deserializes state with `dill`
   - Provides `llm_query()` and `llm_query_batched()` that POST to broker
   - Captures stdout/stderr

**State Persistence:**
```python
# State saved between code blocks in /tmp/rlm_state.dill
{
    "context": <payload>,
    "user_variable_1": <value>,
    "user_variable_2": <value>,
    # ... all user-created variables
}
```

**Setup:**
```python
rlm = RLM(
    environment="modal",
    environment_kwargs={
        "image": custom_modal_image,  # Optional
        "timeout": 600,                # Sandbox timeout
    }
)
```

#### PrimeREPL & DaytonaREPL

Similar architecture to ModalREPL using HTTP broker pattern:
- **PrimeREPL**: Uses Prime Intellect sandboxes
- **DaytonaREPL**: Uses Daytona development environments

Both provide isolated execution with tunnel-based communication.

#### DockerREPL (`docker_repl.py`)

Runs REPL in a Docker container (local machine, but isolated process).

**Features:**
- Uses `python:3.11-slim` image by default (customizable)
- Starts container with exposed port for communication
- State persistence via volume mounts
- Direct socket communication like LocalREPL

---

### LM Clients (`rlm/clients/`)

All clients inherit from `BaseLM` and implement:

```python
class BaseLM:
    def completion(self, prompt: str | dict) -> str:
        """Synchronous completion"""
        
    async def acompletion(self, prompt: str | dict) -> str:
        """Async completion (for batching)"""
        
    def get_usage_summary(self) -> UsageSummary:
        """Total usage across all calls"""
        
    def get_last_usage(self) -> ModelUsageSummary:
        """Usage for the most recent call"""
```

**Supported Clients:**
- `OpenAIClient` - OpenAI API (+ vLLM with OpenAI-compatible endpoint)
- `AnthropicClient` - Anthropic API (Claude models)
- `BedrockClient` - AWS Bedrock (Claude, Llama, Mistral, Titan via Converse API)
- `GeminiClient` - Google Gemini API
- `AzureOpenAIClient` - Azure OpenAI Service
- `LiteLLMClient` - LiteLLM router (100+ models)
- `PortkeyClient` - Portkey AI Gateway

**Usage Tracking:**
Each client maintains:
```python
{
    "model_name": {
        "total_calls": int,
        "total_input_tokens": int,
        "total_output_tokens": int
    }
}
```

#### BedrockClient Details

**AWS Bedrock** provides access to foundation models via a unified API. The `BedrockClient` uses the Converse API for model-agnostic inference.

**Supported Models:**
- **Anthropic Claude**: `us.anthropic.claude-3-5-sonnet-20241022-v2:0`, `anthropic.claude-3-5-haiku-20241022-v1:0`
- **Meta Llama**: `meta.llama3-70b-instruct-v1:0`, `meta.llama3-8b-instruct-v1:0`
- **Mistral AI**: `mistral.mistral-large-2402-v1:0`, `mistral.mixtral-8x7b-instruct-v0:1`
- **Amazon Titan**: `amazon.titan-text-express-v1`, `amazon.titan-text-lite-v1`

**Authentication Methods:**
1. **IAM Role** (recommended for EC2/Lambda): Automatic credential detection
2. **AWS CLI**: Uses credentials from `~/.aws/credentials`
3. **Environment Variables**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
4. **Explicit Parameters**: Pass credentials directly to `BedrockClient`

**Configuration:**
```python
from rlm import RLM
from rlm.clients import BedrockClient

# Method 1: IAM role or AWS CLI (recommended)
rlm = RLM(
    backend="bedrock",
    backend_kwargs={
        "model_name": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "region_name": "us-east-1",
        "max_tokens": 32768,
        "temperature": 1.0,
        "top_p": 0.999,
    }
)

# Method 2: Explicit credentials
import os
rlm = RLM(
    backend="bedrock",
    backend_kwargs={
        "model_name": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "region_name": "us-east-1",
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "max_tokens": 32768,
    }
)
```

**Advantages:**
- **Cost Savings**: Often 20-50% cheaper than direct API access for same models
- **Unified API**: Access multiple model providers through single interface
- **Enterprise Features**: AWS IAM integration, CloudWatch logging, VPC support
- **No API Key Management**: Uses AWS credentials instead of model-specific API keys

**Requirements:**
- `boto3 >= 1.34.0` (Converse API support)
- AWS account with Bedrock access enabled
- Model access granted in AWS Bedrock console (per-region, per-model)

---

### Message History & Prompting

#### System Prompt Structure

```python
[
    {
        "role": "system",
        "content": RLM_SYSTEM_PROMPT  # Instructions for using REPL, llm_query, etc.
    },
    {
        "role": "user",
        "content": "IMPORTANT METADATA:\n"
                   "- Context type: <type>\n"
                   "- Context size: <size>\n"
                   "- Available contexts: context_0, context_1, ...\n"
                   "- Available histories: history_0, history_1, ...\n"
                   "- Iteration: <N>/30\n\n"
                   "USER QUERY: <user_prompt>"
    }
]
```

#### Iteration History Updates

After each iteration, the message history grows:

```python
# Iteration 1
[system, user_query]
→ LM response: "Let me check the context..."
   + ```repl code block```
→ Execute code → output
→ Append to history:
[
    system,
    user_query,
    {"role": "assistant", "content": "Let me check..."},
    {"role": "user", "content": "Code executed:\n```python\n...\n```\n\nREPL output:\n..."}
]

# Iteration 2  
[system, user_query, assistant_1, code_result_1, <iteration_metadata>]
→ LM response with more code...
→ Execute → append
...
```

**Truncation:**
- REPL outputs exceeding 20,000 chars are truncated
- Keeps first 20K characters + indicator: `... + [<N> chars...]`

---

### Types (`rlm/core/types.py`)

**Core Data Structures:**

```python
@dataclass
class REPLResult:
    """Result of executing code in REPL"""
    stdout: str                           # Captured print() output
    stderr: str                           # Captured errors/warnings
    error: str | None                     # Exception if any
    llm_calls: list[RLMChatCompletion]   # Sub-LM calls made during execution

@dataclass
class CodeBlock:
    """A single code block with its execution result"""
    code: str                            # The code string
    result: REPLResult                   # Execution outcome

@dataclass
class RLMIteration:
    """One iteration of the RLM loop"""
    prompt: str | list[dict]             # Input to LM
    response: str                        # LM's text response
    code_blocks: list[CodeBlock]         # Executed code blocks
    iteration_time: float                # Seconds taken
    final_answer: str | None = None      # If FINAL() found

@dataclass
class RLMChatCompletion:
    """Final result of an RLM completion"""
    root_model: str                      # Model name
    prompt: str | dict                   # Original user prompt
    response: str                        # Final answer
    usage_summary: UsageSummary          # Token usage
    execution_time: float                # Total seconds

@dataclass
class RLMMetadata:
    """Metadata logged at start of completion"""
    root_model: str
    max_depth: int
    max_iterations: int
    backend: str
    backend_kwargs: dict
    environment_type: str
    environment_kwargs: dict
    other_backends: list[str] | None
```

---

## Communication Patterns

### Pattern 1: Non-Isolated Environment (LocalREPL, DockerREPL)

Direct TCP socket communication:

```
┌──────────────────────────────────────────────────────────────┐
│  RLM Process                                                 │
│  ┌────────────┐         Socket          ┌────────────────┐   │
│  │    RLM     │◄─────────────────────────│  LMHandler    │   │
│  │ (main loop)│                          │  (TCP Server) │   │
│  └────────────┘                          └────────────────┘   │
│        │                                         ▲            │
│        ▼                                         │            │
│  ┌────────────┐         Socket                  │            │
│  │ LocalREPL  │─────────────────────────────────┘            │
│  │ exec(code) │  llm_query() → send_lm_request()             │
│  └────────────┘                                              │
└──────────────────────────────────────────────────────────────┘
```

**Flow:**
1. LocalREPL executes user code via `exec()`
2. Code calls `llm_query("prompt")`
3. `llm_query()` → `send_lm_request(address, LMRequest)`
4. Opens TCP connection to LMHandler at `(host, port)`
5. Sends length-prefixed JSON request
6. LMHandler processes via `LMRequestHandler.handle()`
7. Returns `LMResponse` with completion
8. `llm_query()` returns response string to code

**Socket Protocol:**
```python
# Send
payload = json.dumps(data).encode("utf-8")
sock.sendall(struct.pack(">I", len(payload)) + payload)

# Receive
length_bytes = sock.recv(4)
length = struct.unpack(">I", length_bytes)[0]
payload = sock.recv(length)
data = json.loads(payload.decode("utf-8"))
```

---

### Pattern 2: Isolated Environment (ModalREPL, PrimeREPL)

HTTP broker pattern with polling:

```
┌───────────────────────────────────────────────────────────────┐
│  Host Process                                                 │
│  ┌──────────┐   Socket   ┌────────────┐   HTTP Poll          │
│  │   RLM    │◄───────────│ LMHandler  │◄──────────────┐      │
│  └──────────┘            └────────────┘               │      │
│                                                        │      │
│  ┌────────────────────────────────────────────────────┴───┐  │
│  │  Poller Thread (background daemon)                     │  │
│  │  while True:                                           │  │
│  │    pending = GET {tunnel_url}/pending                  │  │
│  │    for req in pending:                                 │  │
│  │      response = send_lm_request(handler, req)          │  │
│  │      POST {tunnel_url}/respond with response           │  │
│  │    sleep(0.1)                                          │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       │ HTTPS (encrypted tunnel)
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  Cloud Sandbox (Modal/Prime/Daytona)                         │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  Broker Server (Flask on localhost:8080)              │    │
│  │  pending_requests = {}  # {id: {request, response, event}} │
│  │                                                       │    │
│  │  POST /enqueue: queue request, wait for response     │    │
│  │  GET  /pending: return list of pending requests      │    │
│  │  POST /respond: deliver response, wake event         │    │
│  └───────────────────────────────────────────────────────┘    │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  Execution Script                                     │    │
│  │  llm_query("...") → POST localhost:8080/enqueue       │    │
│  │  (blocks until poller delivers response)              │    │
│  └───────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
```

**Flow:**
1. Sandbox code calls `llm_query("prompt")`
2. `llm_query()` POSTs to `http://localhost:8080/enqueue`
3. Broker queues request, creates `threading.Event`, **blocks**
4. Host poller GETs `/pending`, sees new request
5. Poller forwards to LMHandler via socket
6. LMHandler returns response
7. Poller POSTs to `/respond` with response
8. Broker unblocks `/enqueue`, returns response to sandbox code

**Why Broker?**
- Cloud sandboxes can't directly connect to host sockets
- Tunnel exposes only HTTP endpoints (security)
- Broker provides synchronous API to sandbox code while host polls async

---

## Data Flow

### Complete End-to-End Flow

```
1. User calls rlm.completion("Analyze this...")
   │
   ├─→ 2. RLM._spawn_completion_context()
   │      │
   │      ├─→ Create LMHandler (TCP server on auto-assigned port)
   │      │   └─→ Register LM clients
   │      │
   │      └─→ Create/reuse Environment
   │          ├─→ Load context as variable
   │          └─→ Inject llm_query() function with handler address
   │
   ├─→ 3. RLM enters iteration loop (max 30 iterations)
   │      │
   │      └─→ For each iteration:
   │          │
   │          ├─→ 4. Build prompt (system + history + metadata)
   │          │
   │          ├─→ 5. LMHandler.completion(prompt)
   │          │      └─→ Call LM client (OpenAI/Anthropic/etc.)
   │          │
   │          ├─→ 6. Parse response for code blocks (```repl```)
   │          │
   │          ├─→ 7. For each code block:
   │          │      │
   │          │      ├─→ Environment.execute_code(code)
   │          │      │      │
   │          │      │      ├─→ exec(code, globals, locals)
   │          │      │      │
   │          │      │      └─→ Code may call llm_query()
   │          │      │             │
   │          │      │             ├─→ Non-isolated: Direct socket to LMHandler
   │          │      │             │
   │          │      │             └─→ Isolated: HTTP POST to broker
   │          │      │                    └─→ Poller forwards to LMHandler
   │          │      │
   │          │      └─→ Return REPLResult (stdout/stderr/llm_calls)
   │          │
   │          ├─→ 8. Check for FINAL(...) or FINAL_VAR(...)
   │          │      │
   │          │      ├─→ If found: return RLMChatCompletion
   │          │      │
   │          │      └─→ If not: append iteration to history, continue
   │          │
   │          └─→ Repeat until final answer or max iterations
   │
   └─→ 9. Cleanup
          ├─→ LMHandler.stop()
          └─→ Environment.cleanup() (if not persistent)
```

### Recursive Sub-Call Flow

When code inside REPL calls `llm_query()`:

```
REPL Code Execution
│
├─→ llm_query("What is the magic number in this chunk?")
│      │
│      ├─→ Create LMRequest(prompt="...", model=None, depth=current_depth+1)
│      │
│      ├─→ Send to LMHandler via socket/HTTP
│      │
│      └─→ LMHandler.get_client(model=None, depth=1)
│             │
│             ├─→ If depth==1 and other_backend_client exists
│             │      └─→ Use other_backend_client (e.g., smaller/faster model)
│             │
│             └─→ Else: use default client
│                    │
│                    └─→ Client.completion(prompt)
│                           └─→ Return response string
│
└─→ llm_query() returns response, code continues
```

**Depth Routing:**
- Root RLM (depth 0) can make sub-calls (depth 1)
- Sub-calls (depth 1) at max_depth=1 act as regular LM (no further recursion)
- Allows using different models for root vs sub-calls

---

## Environment Types

### Comparison Table

| Feature | LocalREPL | DockerREPL | ModalREPL | PrimeREPL | DaytonaREPL |
|---------|-----------|------------|-----------|-----------|-------------|
| **Isolation** | Same process | Container | Cloud VM | Cloud VM | Dev env |
| **Security** | Sandboxed builtins | Process isolation | Complete | Complete | Complete |
| **Setup** | None | Docker required | Modal auth | Prime API key | Daytona setup |
| **Speed** | Fastest | Fast | Slower | Slower | Moderate |
| **Cost** | Free | Free | Modal charges | Prime charges | Depends |
| **Communication** | Direct socket | Direct socket | HTTP broker | HTTP broker | HTTP broker |
| **Persistent** | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No |
| **State** | In-memory | Volume mount | dill serialization | dill serialization | dill |
| **Dependencies** | Shared with host | Custom image | Custom image | Custom image | Custom env |

### When to Use Each

**LocalREPL:**
- ✅ Development, testing, benchmarking
- ✅ Trusted code generation
- ✅ Multi-turn conversations (persistent mode)
- ✅ Fast iteration
- ❌ Production with untrusted prompts
- ❌ Need absolute isolation

**DockerREPL:**
- ✅ Better isolation than local
- ✅ Custom dependencies
- ✅ Local development
- ❌ Not as isolated as cloud sandboxes
- ❌ Requires Docker installed

**ModalREPL/PrimeREPL/DaytonaREPL:**
- ✅ Production deployments
- ✅ Untrusted code generation
- ✅ Complete isolation required
- ✅ Custom compute requirements
- ❌ Higher latency
- ❌ Cost considerations

---

## Extension Points

### Adding a New LM Client

**Reference Implementation**: See [rlm/clients/bedrock.py](rlm/clients/bedrock.py) for a complete example.

1. **Inherit from BaseLM:**
   ```python
   from rlm.clients.base_lm import BaseLM
   from rlm.core.types import ModelUsageSummary, UsageSummary
   from collections import defaultdict
   
   class MyClient(BaseLM):
       def __init__(self, api_key: str, model_name: str, **kwargs):
           super().__init__(model_name=model_name, **kwargs)
           # Initialize client (e.g., SDK, HTTP client)
           self.client = SomeSDK(api_key=api_key)
           
           # Track usage per model
           self.model_call_counts: dict[str, int] = defaultdict(int)
           self.model_input_tokens: dict[str, int] = defaultdict(int)
           self.model_output_tokens: dict[str, int] = defaultdict(int)
           
       def completion(self, prompt: str | dict) -> str:
           # Convert prompt format if needed
           messages = self._prepare_messages(prompt)
           
           # Make API call
           response = self.client.create_completion(messages=messages)
           
           # Track usage
           self._track_usage(response)
           
           # Return text response
           return response.text
           
       async def acompletion(self, prompt: str | dict) -> str:
           # Implement async (or wrap sync if SDK doesn't support async)
           return self.completion(prompt)
           
       def get_usage_summary(self) -> UsageSummary:
           # Return aggregated usage across all models
           model_summaries = {}
           for model in self.model_call_counts:
               model_summaries[model] = ModelUsageSummary(
                   total_calls=self.model_call_counts[model],
                   total_input_tokens=self.model_input_tokens[model],
                   total_output_tokens=self.model_output_tokens[model],
               )
           return UsageSummary(model_usage_summaries=model_summaries)
           
       def get_last_usage(self) -> ModelUsageSummary:
           # Return last call usage
           return ModelUsageSummary(
               total_calls=1,
               total_input_tokens=self.last_prompt_tokens,
               total_output_tokens=self.last_completion_tokens,
           )
       
       def _prepare_messages(self, prompt: str | dict) -> list:
           # Convert RLM message format to your API's format
           if isinstance(prompt, str):
               return [{"role": "user", "content": prompt}]
           # Handle message list format...
           
       def _track_usage(self, response):
           # Extract and track token usage from response
           self.model_call_counts[model] += 1
           self.model_input_tokens[model] += response.input_tokens
           # ...
   ```

2. **Add to ClientBackend type in `rlm/core/types.py`:**
   ```python
   ClientBackend = Literal[
       "openai",
       # ... existing backends
       "my_client",  # NEW
   ]
   ```

3. **Register in `rlm/clients/__init__.py`:**
   ```python
   def get_client(backend: ClientBackend, kwargs: dict) -> BaseLM:
       if backend == "my_client":
           from rlm.clients.my_client import MyClient
           return MyClient(**kwargs)
       # ...
       else:
           raise ValueError(
               f"Unknown backend: {backend}. Supported backends: "
               f"['openai', ..., 'my_client']"
           )
   ```

4. **Create example in `examples/my_client_example.py`**

5. **Update documentation** - Add to README.md and ARCHITECTURE.md

**Key Principles** (from AGENTS.md):
- Use `defaultdict` for usage tracking (see `BedrockClient` pattern)
- Support both `str` and `list[dict]` prompt formats
- Extract system messages if your API handles them separately
- No defensive programming - fail fast with clear error messages
- Minimal dependencies - use optional extras if needed: `pip install -e ".[my_client]"`

---

### Adding a New Environment

1. **Choose base class:**
   - `NonIsolatedEnv` for same-machine execution
   - `IsolatedEnv` for cloud sandboxes

2. **Implement required methods:**
   ```python
   from rlm.environments.base_env import NonIsolatedEnv
   
   class MyEnvironment(NonIsolatedEnv):
       def __init__(self, lm_handler_address, context_payload, **kwargs):
           super().__init__(**kwargs)
           self.lm_handler_address = lm_handler_address
           self.setup()
           if context_payload:
               self.load_context(context_payload)
           
       def setup(self):
           # Initialize execution namespace
           # Inject llm_query(), FINAL_VAR(), etc.
           
       def load_context(self, context_payload):
           # Make context available to code
           
       def execute_code(self, code: str) -> REPLResult:
           # Execute code, return result
           
       def cleanup(self):
           # Clean up resources
   ```

3. **For isolated environments:**
   - Implement HTTP broker (see ModalREPL)
   - Set up polling thread
   - Handle state serialization

4. **Register in `rlm/environments/__init__.py`:**
   ```python
   def get_environment(env_type: EnvironmentType, kwargs: dict) -> BaseEnv:
       if env_type == "my_env":
           return MyEnvironment(**kwargs)
       # ...
   ```

---

### Multi-Turn Persistence Support

To add persistence to an environment, implement the `SupportsPersistence` protocol:

```python
from rlm.environments.base_env import SupportsPersistence

class MyPersistentEnv(NonIsolatedEnv, SupportsPersistence):
    def __init__(self, ...):
        super().__init__(persistent=True, ...)
        self._context_count = 0
        self._history_count = 0
        
    def update_handler_address(self, address: tuple[str, int]):
        """Update LM handler address for new completion"""
        self.lm_handler_address = address
        
    def add_context(self, context_payload, context_index=None) -> int:
        """Add context_N variable"""
        if context_index is None:
            context_index = self._context_count
        self.globals[f"context_{context_index}"] = context_payload
        if context_index == 0:
            self.globals["context"] = context_payload  # Alias
        self._context_count = max(self._context_count, context_index + 1)
        return context_index
        
    def get_context_count(self) -> int:
        """Return number of contexts"""
        return self._context_count
        
    def add_history(self, messages, history_index=None) -> int:
        """Add history_N variable"""
        if history_index is None:
            history_index = self._history_count
        self.globals[f"history_{history_index}"] = messages
        if history_index == 0:
            self.globals["history"] = messages  # Alias
        self._history_count = max(self._history_count, history_index + 1)
        return history_index
        
    def get_history_count(self) -> int:
        """Return number of histories"""
        return self._history_count
```

**Test your implementation:**
```bash
uv run pytest tests/test_local_repl_persistent.py -v
```

---

### Custom System Prompts

Override the default RLM system prompt:

```python
CUSTOM_PROMPT = """
You are an expert at analyzing documents.
Use the REPL environment to process the context variable.
Available functions: llm_query(), FINAL_VAR(), SHOW_VARS()
...
"""

rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4"},
    custom_system_prompt=CUSTOM_PROMPT
)
```

---

### Logging & Visualization

**Enable logging:**
```python
from rlm.logger import RLMLogger

logger = RLMLogger(log_dir="./logs")
rlm = RLM(..., logger=logger)
```

**Log format** (`.jsonl`):
```json
{"type": "metadata", "data": {...}}
{"type": "iteration", "iteration": 1, "data": {...}}
{"type": "iteration", "iteration": 2, "data": {...}}
```

**Visualize trajectories:**
```bash
cd visualizer
npm install
npm run dev
# Upload .jsonl file in UI
```

---

## Key Takeaways

1. **RLM = Iterative LM + REPL + Recursive Queries**
   - Not just "more context" - it's programmatic context examination

2. **Environments provide isolation levels**
   - Local for speed, cloud for security

3. **LM Handler is the communication hub**
   - Multi-threaded socket server
   - Routes all sub-LM queries

4. **Two communication patterns**
   - Direct socket (non-isolated)
   - HTTP broker with polling (isolated)

5. **Extensible by design**
   - Add clients: inherit `BaseLM`
   - Add environments: inherit `BaseEnv`
   - Add persistence: implement `SupportsPersistence`

6. **Depth controls recursion**
   - Depth 0: Root RLM
   - Depth 1: Sub-calls (can use different model)
   - At max_depth: Fallback to regular LM

7. **Message history grows iteratively**
   - System prompt + user query
   - + assistant response
   - + code execution results
   - + assistant response
   - + ...
   - Until FINAL() or max iterations

---

## Additional Resources

- **Paper:** [Recursive Language Models (arXiv)](https://arxiv.org/abs/2512.24601)
- **Blog:** [RLM Blogpost](https://alexzhang13.github.io/blog/2025/rlm/)
- **Documentation:** [Official Docs](https://alexzhang13.github.io/rlm/)
- **Contributing:** See [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md)
- **Examples:** See `examples/` directory

---

*Last Updated: 2025*
