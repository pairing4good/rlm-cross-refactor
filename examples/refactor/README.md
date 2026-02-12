# RLM Refactor - Interactive Code Refactoring

Use the built-in RLM refactoring CLI for code analysis and transformation tasks with YAML-based configuration and Docker isolation.

## Quick Start

### Try the Sample Example

Run the included sample that searches for "bob" across multiple repositories:

```bash
# From the repository root
uv run python -m rlm.refactor.cli --working-dir examples/refactor/sample-working

# Or interactive mode (will prompt for directory):
uv run python -m rlm.refactor.cli
# Enter: examples/refactor/sample-working
```

That's it! The sample includes:
- `rlm-instructions.md` - Task: Find which file contains "bob" across all repos
- `rlm-config.yaml` - Bedrock configuration with Sonnet (root) + Haiku (sub) multi-tier setup
- Three git repositories with 10 text files total (one contains "bob")
  - `sample-repo-one/` - 2 files
  - `sample-repo-two/` - 3 files  
  - `sample-repo-three/` - 5 files (contains the target)

**Expected output**: The RLM will search across all repositories and report that `sample-repo-three/file_05.txt` contains "bob".

**Prerequisites**: 
- AWS credentials configured (for Bedrock backend)
- Docker installed and running (for safe code execution isolation)
- Run `uv sync` to install dependencies

### Create Your Own Task

1. **Create a working directory** with your task:
   ```bash
   mkdir my-refactor-task
   cd my-refactor-task
   ```

2. **Create required files**:
   
   **`rlm-instructions.md`** (task description):
   ```markdown
   # Task: Find Bob
   
   Search through all `.txt` files and identify which file contains the word "bob".
   Return only the filename.
   ```
   
   **`rlm-config.yaml`** (configuration):
   ```yaml
   backend: bedrock
   backend_kwargs:
     model_name: us.anthropic.claude-3-5-sonnet-20241022-v2:0
     region_name: us-east-1
   
   max_root_tokens: 50000
   max_sub_tokens: 25000
   ```

3. **Run the refactoring CLI**:
   ```bash
   # From repository root
   uv run python -m rlm.refactor.cli --working-dir ./my-refactor-task
   
   # Or use interactive mode
   uv run python -m rlm.refactor.cli
   # Enter: ./my-refactor-task
   ```

## Configuration Requirements

Both `rlm-instructions.md` and `rlm-config.yaml` are **required** in your working directory.

### Required: `rlm-config.yaml`

Minimal configuration:

```yaml
# Minimal configuration
backend: bedrock
backend_kwargs:
  model_name: us.anthropic.claude-3-5-sonnet-20241022-v2:0
  region_name: us-east-1
  max_tokens: 8000
```

See [rlm-config.yaml](./rlm-config.yaml) for a complete template with all options documented.

### Configuration Reference

#### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `backend` | string | LM provider | `bedrock`, `openai`, `anthropic` |
| `backend_kwargs.model_name` | string | Model identifier | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` |

#### Common Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend_kwargs.max_tokens` | int | 8000 | Max tokens per response (≥100) |
| `backend_kwargs.temperature` | float | 1.0 | Sampling temperature (0.0-2.0) |
| `backend_kwargs.region_name` | string | `us-east-1` | AWS region (Bedrock only) |
| `max_root_tokens` | int | 100000 | Session-wide token budget |
| `max_iterations` | int | 50 | Max REPL iterations |
| `environment` | string | `local` | Execution environment |
| `verbose` | bool | true | Enable rich console output |
| `log_dir` | string | `rlm-logs` | Log directory (relative path) |

#### Multi-Tier Configuration (Cost Optimization)

Use a cheaper sub-model for simple tasks called via `llm_query()`:

```yaml
backend: bedrock
backend_kwargs:
  model_name: us.anthropic.claude-3-5-sonnet-20241022-v2:0  # Root model
  region_name: us-east-1

# Enable multi-tier routing
other_backends:
  - bedrock
other_backend_kwargs:
  - model_name: us.anthropic.claude-3-5-haiku-20241022-v1:0  # Sub model
    region_name: us-east-1

max_root_tokens: 100000
max_sub_tokens: 50000
```

**Cost savings**: ~80-90% on sub-queries (Haiku is 10x cheaper than Sonnet)

## Supported Backends & Models

### AWS Bedrock

```yaml
backend: bedrock
backend_kwargs:
  model_name: us.anthropic.claude-3-5-sonnet-20241022-v2:0
  region_name: us-east-1
```

**Popular models**:
- `us.anthropic.claude-3-5-sonnet-20241022-v2:0` - Best quality
- `us.anthropic.claude-3-5-haiku-20241022-v1:0` - Fast & cheap
- `meta.llama3-70b-instruct-v1:0` - Open source

**Credentials**: AWS CLI config or environment variables

### OpenAI

```yaml
backend: openai
backend_kwargs:
  model_name: gpt-4o
  # api_key: auto-detected from OPENAI_API_KEY
```

**Popular models**:
- `gpt-4o` - Best quality
- `gpt-4o-mini` - Fast & cheap
- `gpt-4-turbo` - Legacy flagship

**Credentials**: `OPENAI_API_KEY` environment variable

### Anthropic (Direct API)

```yaml
backend: anthropic
backend_kwargs:
  model_name: claude-sonnet-4-20250514
  # api_key: auto-detected from ANTHROPIC_API_KEY
```

**Popular models**:
- `claude-sonnet-4-20250514` - Latest Sonnet
- `claude-3-5-sonnet-20241022` - Previous generation
- `claude-3-5-haiku-20241022` - Fast & cheap

**Credentials**: `ANTHROPIC_API_KEY` environment variable

### Google Gemini

```yaml
backend: gemini
backend_kwargs:
  model_name: gemini-2.0-flash-exp
  # api_key: auto-detected from GEMINI_API_KEY
```

**Popular models**:
- `gemini-2.0-flash-exp` - Very cheap (experimental)
- `gemini-1.5-pro` - Best quality
- `gemini-1.5-flash` - Fast

**Credentials**: `GEMINI_API_KEY` environment variable

## Execution Environment

The CLI **hardcodes Docker** for safety and isolation. All refactoring code executes inside a `python:3.12-slim` Docker container with the working directory mounted.

**Why Docker?**
- ✅ **Safety**: Prevents accidental damage to your host system
- ✅ **Isolation**: Clean environment for each run
- ✅ **Reproducibility**: Consistent Python version across systems

**Requirements**:
- Docker installed and running
- Working directory is mounted read-write into container at `/workspace`

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'boto3'" (or other missing modules)

**Cause**: Running `python -m rlm.refactor.cli` directly doesn't use the project's dependencies.

**Solution**: Use `uv run` to execute within the virtual environment:
```bash
# Wrong - dependencies not available
python -m rlm.refactor.cli --working-dir ./my-task

# Correct - uses project dependencies
uv run python -m rlm.refactor.cli --working-dir ./my-task
```

**Alternative**: Install missing backend-specific dependencies:
```bash
# For Bedrock (requires boto3)
uv pip install boto3

# Or install all optional dependencies
uv sync --all-extras
```

### Error: "Missing required file: rlm-instructions.md"

**Solution**: Create `rlm-instructions.md` in your working directory:
```bash
cat > my-dir/rlm-instructions.md << 'EOF'
# My Task

Task description here
EOF
```

### Error: "Missing required file: rlm-config.yaml"

**Solution**: Create `rlm-config.yaml` in your working directory:
```bash
cat > my-dir/rlm-config.yaml << 'EOF'
backend: bedrock
backend_kwargs:
  model_name: us.anthropic.claude-3-5-sonnet-20241022-v2:0
  region_name: us-east-1
max_root_tokens: 50000
EOF
```

### Error: "Failed to parse rlm-config.yaml"

**Common issues**:
- Indentation must use spaces (not tabs)
- Missing quotes around special characters
- Incorrect YAML syntax

**Solution**: Check YAML syntax or remove config file to use defaults

### Error: "Field 'backend_kwargs.max_tokens' = 50 is too low"

**Solution**: Increase `max_tokens` to at least 100:
```yaml
backend_kwargs:
  max_tokens: 4096  # Must be >= 100
```

### Error: "Field 'max_root_tokens' should not be in 'backend_kwargs'"

**Common mistake**: Token budgets are top-level fields, not backend kwargs.

**Wrong**:
```yaml
backend_kwargs:
  model_name: gpt-4o
  max_root_tokens: 100000  # ❌ Wrong location
```

**Correct**:
```yaml
backend_kwargs:
  model_name: gpt-4o

max_root_tokens: 100000  # ✅ Top-level field
```

### Warning: "AWS credentials not detected for Bedrock"

**Solution**: Set AWS credentials:
```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Option 2: AWS profile
export AWS_PROFILE=my-profile

# Option 3: AWS CLI config
aws configure
```

### Error: Docker container failures or image issues

**Symptoms**:
- Container won't start
- Git installation failures
- Execution errors inside Docker

**Solution**: Check the Docker diagnostic logs in `{log_dir}/docker_*.jsonl`:
```bash
# Find errors in Docker logs
cat my-refactor-task/rlm-logs/docker_*.jsonl | jq 'select(.level == "ERROR" or .level == "WARNING")'

# Check container startup
cat my-refactor-task/rlm-logs/docker_*.jsonl | jq 'select(.message | contains("Container"))'

# View git installation details
cat my-refactor-task/rlm-logs/docker_*.jsonl | jq 'select(.message | contains("Git"))'
```

Common Docker issues:
- **Image pull fails**: Check your internet connection and Docker daemon status
- **Volume mount errors**: Verify the working directory path exists and is accessible
- **Git install fails**: The container needs internet access for apt-get; check network settings
- **Permission issues**: Ensure Docker has permissions to access the working directory

### Question: "Where can I see what prompt was sent to the LLM?"

Check the log files in `{working_dir}/rlm_logs/*.jsonl` (or custom `log_dir` from config):

```bash
# View logs
cat my-refactor-task/rlm_logs/refactor_*.jsonl | jq .

# See the exact messages
cat my-refactor-task/rlm_logs/refactor_*.jsonl | jq '.messages'
```

The CLI sends your `rlm-instructions.md` content directly to the LLM along with the refactoring system prompt.

## Cost Optimization

### Use Multi-Tier Routing

**Single-tier** (all Sonnet):
```yaml
backend: bedrock
backend_kwargs:
  model_name: us.anthropic.claude-3-5-sonnet-20241022-v2:0
# Cost: ~$9 for complex task with many sub-queries
```

**Multi-tier** (Sonnet + Haiku):
```yaml
backend: bedrock
backend_kwargs:
  model_name: us.anthropic.claude-3-5-sonnet-20241022-v2:0
other_backends:
  - bedrock
other_backend_kwargs:
  - model_name: us.anthropic.claude-3-5-haiku-20241022-v1:0
# Cost: ~$1.50 for same task (83% savings!)
```

### Cost Reference (2026)

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| **Claude 3.5 Sonnet** | $3/M | $15/M | Best quality, root model |
| **Claude 3.5 Haiku** | $0.25/M | $1.25/M | Fast, cheap sub-model |
| **GPT-4o** | $2.50/M | $10/M | High quality |
| **GPT-4o-mini** | $0.15/M | $0.60/M | Budget sub-model |
| **Gemini 2.0 Flash** | $0.10/M | $0.30/M | Very cheap |

**Recommendation**: Use premium models (Sonnet, GPT-4o) for root, budget models (Haiku, mini) for subs.

## Advanced Usage

### Custom System Prompt

The refactoring CLI uses the `REFACTOR_SYSTEM_PROMPT` from `rlm/refactor/prompts.py`. To customize, modify the CLI source:

```python
# In rlm/refactor/cli.py (line ~245)
rlm = RLM(
    backend=backend,
    backend_kwargs=backend_kwargs,
    environment="docker",
    environment_kwargs=environment_kwargs,
    custom_system_prompt=REFACTOR_SYSTEM_PROMPT,  # ← Modify this
    # ...
)
```

### Persistent Environments

For multiple related tasks, use persistent environments:

```yaml
# Not yet supported in config, use Python API:
# rlm = RLM(..., persistent=True)
```

### Logging & Analysis

Two types of logs are written to `{log_dir}` (default: `rlm-logs/`):

**1. Refactor trajectory logs** (`refactor_*.jsonl`):
```bash
# View execution traces
cat my-refactor-task/rlm-logs/refactor_*.jsonl | jq .

# Analyze token usage
cat my-refactor-task/rlm-logs/refactor_*.jsonl | jq '.usage'

# Examine specific iteration
cat my-refactor-task/rlm-logs/refactor_*.jsonl | jq 'select(.iteration == 3)'
```

**2. Docker diagnostic logs** (`docker_*.jsonl`):
Tracks container lifecycle events (setup, execution errors, cleanup):
```bash
# View Docker events
cat my-refactor-task/rlm-logs/docker_*.jsonl | jq .

# Check for errors
cat my-refactor-task/rlm-logs/docker_*.jsonl | jq 'select(.level == "ERROR")'

# Monitor git installation
cat my-refactor-task/rlm-logs/docker_*.jsonl | jq 'select(.message | contains("Git"))'
```

**Log correlation**: Both logs share the same timestamp/run_id suffix for easy matching.

## Files

```
examples/refactor/
├── README.md              # This file
├── sample-working/        # Sample working directory (ready to run!)
│   ├── rlm-config.yaml        # Example Bedrock configuration
│   ├── rlm-instructions.md    # Task: Find which file contains "bob"
│   ├── sample-repo-one/       # Git repository #1
│   │   ├── .git/
│   │   ├── file_01.txt
│   │   └── file_02.txt
│   ├── sample-repo-two/       # Git repository #2
│   │   ├── .git/
│   │   ├── file_08.txt
│   │   ├── file_09.txt
│   │   └── file_10.txt
│   └── sample-repo-three/     # Git repository #3
│       ├── .git/
│       ├── file_03.txt
│       ├── file_04.txt
│       ├── file_05.txt        # ← Contains "bob"
│       ├── file_06.txt
│       └── file_07.txt
└── rlm-config.yaml        # Configuration template with all options
```

**CLI Location**: `rlm/refactor/cli.py` (run via `python -m rlm.refactor.cli`)

## See Also

- [RLM Documentation](../../README.md)
- [Multi-tier example](../multi_tier_example.py)
- [Bedrock example](../bedrock_example.py)
- [Refactoring CLI source](../../rlm/refactor/cli.py)
