"""
CLI entry point for rlm-refactor.

Reads configuration from rlm-config.yaml and task instructions from
rlm-instructions.md in the working directory. Executes refactoring in
an isolated Docker container for safety.

Configuration:
    Working Directory (required):
        - Contains rlm-config.yaml (model configuration)
        - Contains rlm-instructions.md (refactoring task description)
        - Mounted into Docker container at /workspace

    rlm-config.yaml:
        - backend: LM provider (openai, anthropic, bedrock, gemini, portkey)
        - backend_kwargs: Model config with model_name (required)
        - other_backends: (optional) Sub-model for cost optimization
        - max_root_tokens: Session-wide token budget for root model
        - max_sub_tokens: Session-wide token budget for sub-model
        - max_iterations: Max REPL iterations (default: 30)
        - verbose: Enable rich console output (default: false)
        - log_dir: Log directory name (default: rlm_logs)

Usage:
    # Interactive mode (prompts for working directory)
    rlm-refactor

    # Direct mode (specify working directory)
    rlm-refactor --working-dir ./my-project

    # Working directory must contain:
    #   - rlm-config.yaml (configuration)
    #   - rlm-instructions.md (task description)

Examples:
    # Setup
    mkdir my-refactoring && cd my-refactoring

    # Create task instructions
    cat > rlm-instructions.md << 'EOF'
    # Find all occurrences of "TODO"
    Search all Python files and list every TODO comment.
    EOF

    # Create config
    cat > rlm-config.yaml << 'EOF'
    backend: bedrock
    backend_kwargs:
      model_name: us.anthropic.claude-3-5-sonnet-20241022-v2:0
      region_name: us-east-1
    max_root_tokens: 50000
    verbose: true
    EOF

    # Run refactoring
    rlm-refactor --working-dir .
"""

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.prompt import Prompt

from rlm.core.rlm import RLM
from rlm.logger import RLMLogger
from rlm.refactor.prompts import REFACTOR_SYSTEM_PROMPT

INSTRUCTIONS_FILENAME = "rlm-instructions.md"
CONFIG_FILENAME = "rlm-config.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rlm-refactor",
        description="Run an RLM-powered code refactoring agent in Docker. "
        "All configuration is read from rlm-config.yaml in the working directory.",
    )
    parser.add_argument(
        "--working-dir",
        default=None,
        help="Working directory containing rlm-config.yaml and rlm-instructions.md "
        "(default: interactive prompt)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    console = Console()
    args = parse_args(argv)

    # Step 1: Get working directory (prompt if not provided)
    if args.working_dir is None:
        console.print("\n[bold cyan]RLM Refactoring Agent[/bold cyan]")
        console.print(
            "All configuration read from [yellow]rlm-config.yaml[/yellow] and [yellow]rlm-instructions.md[/yellow]\n"
        )

        while True:
            working_dir_input = Prompt.ask(
                "[bold cyan]Enter working directory path[/bold cyan]", default=str(Path.cwd())
            )
            working_dir = Path(working_dir_input).expanduser().resolve()

            if not working_dir.exists():
                console.print(f"[red]✗ Directory does not exist: {working_dir}[/red]\n")
                continue
            if not working_dir.is_dir():
                console.print(f"[red]✗ Path is not a directory: {working_dir}[/red]\n")
                continue
            break
    else:
        working_dir = Path(args.working_dir).expanduser().resolve()
        if not working_dir.exists():
            console.print(f"[red]Error: Working directory does not exist: {working_dir}[/red]")
            sys.exit(1)
        if not working_dir.is_dir():
            console.print(f"[red]Error: Path is not a directory: {working_dir}[/red]")
            sys.exit(1)

    # Step 2: Load rlm-instructions.md
    instructions_path = working_dir / INSTRUCTIONS_FILENAME
    if not instructions_path.exists():
        console.print(
            f"[red]Error: {INSTRUCTIONS_FILENAME} not found in {working_dir}[/red]\n"
            f"Create {INSTRUCTIONS_FILENAME} with your refactoring task description."
        )
        sys.exit(1)

    instructions = instructions_path.read_text()
    if not instructions.strip():
        console.print(f"[red]Error: {INSTRUCTIONS_FILENAME} is empty[/red]")
        sys.exit(1)

    # Step 3: Load rlm-config.yaml
    config_path = working_dir / CONFIG_FILENAME
    if not config_path.exists():
        console.print(
            f"[red]Error: {CONFIG_FILENAME} not found in {working_dir}[/red]\n"
            f"Create {CONFIG_FILENAME} with your model and backend configuration."
        )
        sys.exit(1)

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        console.print(f"[red]Error: Invalid YAML in {CONFIG_FILENAME}:[/red]\n{e}")
        sys.exit(1)

    # Step 4: Extract and validate configuration
    backend = config.get("backend")
    if not backend:
        console.print(
            "[red]Error: 'backend' is required in config (e.g., openai, anthropic, bedrock)[/red]"
        )
        sys.exit(1)

    backend_kwargs = config.get("backend_kwargs", {})
    if not isinstance(backend_kwargs, dict):
        console.print("[red]Error: 'backend_kwargs' must be a dictionary[/red]")
        sys.exit(1)
    if "model_name" not in backend_kwargs:
        console.print("[red]Error: 'backend_kwargs.model_name' is required[/red]")
        sys.exit(1)

    # Validate forbidden parameters
    forbidden = ["max_root_tokens", "max_sub_tokens"]
    for param in forbidden:
        if param in backend_kwargs:
            console.print(
                f"[red]Error: '{param}' should be a top-level config parameter, "
                f"not inside backend_kwargs[/red]"
            )
            sys.exit(1)

    # Handle multi-tier configuration
    other_backends = config.get("other_backends")
    other_backend_kwargs = config.get("other_backend_kwargs")

    if other_backends is not None or other_backend_kwargs is not None:
        if other_backends is None or other_backend_kwargs is None:
            console.print(
                "[red]Error: Both 'other_backends' and 'other_backend_kwargs' must be specified together[/red]"
            )
            sys.exit(1)
        if not isinstance(other_backends, list) or not isinstance(other_backend_kwargs, list):
            console.print(
                "[red]Error: 'other_backends' and 'other_backend_kwargs' must be lists[/red]"
            )
            sys.exit(1)
        if len(other_backends) != len(other_backend_kwargs):
            console.print(
                "[red]Error: 'other_backends' and 'other_backend_kwargs' must have same length[/red]"
            )
            sys.exit(1)

        # Validate sub-backend kwargs
        for i, sub_kwargs in enumerate(other_backend_kwargs):
            if not isinstance(sub_kwargs, dict):
                console.print(f"[red]Error: other_backend_kwargs[{i}] must be a dictionary[/red]")
                sys.exit(1)
            for param in forbidden:
                if param in sub_kwargs:
                    console.print(
                        f"[red]Error: '{param}' should be a top-level config parameter, "
                        f"not inside other_backend_kwargs[{i}][/red]"
                    )
                    sys.exit(1)

    # Extract token budgets
    max_root_tokens = config.get("max_root_tokens", 1_000_000)
    max_sub_tokens = config.get("max_sub_tokens", 1_000_000)

    if not isinstance(max_root_tokens, int) or max_root_tokens <= 0:
        console.print("[red]Error: 'max_root_tokens' must be a positive integer[/red]")
        sys.exit(1)
    if not isinstance(max_sub_tokens, int) or max_sub_tokens <= 0:
        console.print("[red]Error: 'max_sub_tokens' must be a positive integer[/red]")
        sys.exit(1)

    # Extract execution settings
    max_iterations = config.get("max_iterations", 30)
    verbose = config.get("verbose", False)
    log_dir_name = config.get("log_dir", "rlm_logs")

    # Step 5: Setup logging
    log_dir = working_dir / log_dir_name
    log_dir.mkdir(exist_ok=True)
    logger = RLMLogger(log_dir=str(log_dir), file_name="refactor")

    # Create Docker diagnostic log with matching timestamp/run_id
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = str(uuid.uuid4())[:8]
    docker_log_path = str(log_dir / f"docker_{timestamp}_{run_id}.jsonl")

    # Step 6: Build environment kwargs
    environment_kwargs = {
        "working_dir": str(working_dir),
        "image": "python:3.12-slim",
        "docker_log_path": docker_log_path,
    }

    # Step 7: Create RLM instance
    console.print(f"\n[green]✓[/green] Working directory: {working_dir}")
    console.print(f"[green]✓[/green] Backend: {backend}")
    console.print(f"[green]✓[/green] Root model: {backend_kwargs.get('model_name')}")
    if other_backends:
        console.print(f"[green]✓[/green] Sub model: {other_backend_kwargs[0].get('model_name')}")
    console.print(f"[green]✓[/green] Max iterations: {max_iterations}")
    console.print(
        f"[green]✓[/green] Token budgets: {max_root_tokens:,} root / {max_sub_tokens:,} sub\n"
    )

    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment="docker",
        environment_kwargs=environment_kwargs,
        custom_system_prompt=REFACTOR_SYSTEM_PROMPT,
        logger=logger,
        max_iterations=max_iterations,
        max_root_tokens=max_root_tokens,
        max_sub_tokens=max_sub_tokens,
        verbose=verbose,
        other_backends=other_backends,
        other_backend_kwargs=other_backend_kwargs,
    )

    # Step 8: Run refactoring
    root_prompt = instructions[:500] if len(instructions) > 500 else instructions

    # Show spinner only when verbose mode is disabled (verbose output provides its own progress)
    if verbose:
        console.print("[bold cyan]→[/bold cyan] Starting RLM agent...\n")
        result = rlm.completion(prompt=instructions, root_prompt=root_prompt)
    else:
        with console.status(
            "[bold cyan]RLM agent iterating through your instructions...[/bold cyan]",
            spinner="dots",
        ):
            result = rlm.completion(prompt=instructions, root_prompt=root_prompt)

    # Step 9: Display results
    console.print("\n" + "=" * 60)
    console.print("[bold green]REFACTORING COMPLETE[/bold green]")
    console.print("=" * 60)
    console.print(f"\n{result.response}")
    console.print(f"\n[dim]Execution time: {result.execution_time:.1f}s[/dim]")
    console.print(f"[dim]Logs written to: {log_dir}[/dim]")


if __name__ == "__main__":
    main()
