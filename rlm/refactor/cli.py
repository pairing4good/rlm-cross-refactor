"""
CLI entry point for rlm-refactor.

Mounts a working directory containing git repos into a Docker container,
reads rlm-instructions.md as the task prompt, and runs the RLM loop
with a refactoring-specific system prompt.

Usage:
    # Basic usage
    rlm-refactor --working-dir ./my-repos --model gpt-4o

    # With inference parameters
    rlm-refactor --working-dir ./my-repos --model gpt-4o \\
      --max-tokens 8000 --temperature 0.7

    # With sub-model configuration
    rlm-refactor --working-dir ./my-repos --model gpt-4o \\
      --sub-model gpt-4o-mini --sub-max-tokens 4000

    # With provider-specific parameters
    rlm-refactor --working-dir ./my-repos --model custom \\
      --backend-arg base_url=http://localhost:8080

Note: --max-tokens is per-response limit (goes to LM backend).
      --max-root-tokens is session-wide budget (goes to RLM).
"""

import argparse
import os
import sys
from typing import Any

from rlm.core.rlm import RLM
from rlm.logger import RLMLogger
from rlm.refactor.prompts import REFACTOR_SYSTEM_PROMPT

INSTRUCTIONS_FILENAME = "rlm-instructions.md"


def parse_arg_value(arg: str) -> tuple[str, Any]:
    """Parse KEY=VALUE string, auto-detecting type."""
    if "=" not in arg:
        raise ValueError(f"Invalid --backend-arg format: '{arg}'. Expected KEY=VALUE")
    key, value_str = arg.split("=", 1)

    # Auto-detect type
    if value_str.lower() in ("true", "false"):
        return key, value_str.lower() == "true"
    try:
        return key, int(value_str)
    except ValueError:
        try:
            return key, float(value_str)
        except ValueError:
            return key, value_str  # String


def build_backend_kwargs(args: argparse.Namespace, is_sub: bool = False) -> dict[str, Any]:
    """Build backend_kwargs from CLI args, handling root vs sub parameters."""
    kwargs: dict[str, Any] = {"model_name": args.sub_model if is_sub else args.model}

    # Common inference parameters
    if not is_sub:
        if args.max_tokens is not None:
            kwargs["max_tokens"] = args.max_tokens
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            kwargs["top_p"] = args.top_p
    else:
        if args.sub_max_tokens is not None:
            kwargs["max_tokens"] = args.sub_max_tokens
        if args.sub_temperature is not None:
            kwargs["temperature"] = args.sub_temperature
        if args.sub_top_p is not None:
            kwargs["top_p"] = args.sub_top_p

    # Parse generic --backend-arg / --sub-backend-arg
    arg_list = args.sub_backend_arg if is_sub else args.backend_arg
    if arg_list:
        for arg in arg_list:
            key, value = parse_arg_value(arg)
            kwargs[key] = value

    return kwargs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rlm-refactor",
        description="Run an RLM-powered code refactoring agent in Docker.",
    )
    parser.add_argument(
        "--working-dir",
        required=True,
        help="Host directory containing git repos and rlm-instructions.md",
    )
    parser.add_argument(
        "--backend",
        default="openai",
        help="Root LM backend (default: openai)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Root model name (e.g., gpt-4o, claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--sub-backend",
        default=None,
        help="Sub-LM backend for llm_query() calls (default: same as --backend)",
    )
    parser.add_argument(
        "--sub-model",
        default=None,
        help="Sub-LM model name for llm_query() calls",
    )

    # Common inference parameters for root model
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Per-response token limit for root model (e.g., 8000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for root model (0.0-2.0, default varies by provider)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Nucleus sampling for root model (0.0-1.0)",
    )

    # Common inference parameters for sub model
    parser.add_argument(
        "--sub-max-tokens",
        type=int,
        help="Per-response token limit for sub model",
    )
    parser.add_argument(
        "--sub-temperature",
        type=float,
        help="Temperature for sub model",
    )
    parser.add_argument(
        "--sub-top-p",
        type=float,
        help="Nucleus sampling for sub model",
    )

    # Generic parameter injection
    parser.add_argument(
        "--backend-arg",
        action="append",
        metavar="KEY=VALUE",
        help="Additional backend parameter (e.g., base_url=http://localhost:8000). Can be repeated.",
    )
    parser.add_argument(
        "--sub-backend-arg",
        action="append",
        metavar="KEY=VALUE",
        help="Additional sub-backend parameter. Can be repeated.",
    )

    # RLM-level token budgets
    parser.add_argument(
        "--max-root-tokens",
        type=int,
        default=1_000_000,
        help="Maximum total tokens for root agent across all iterations (default: 1M)",
    )
    parser.add_argument(
        "--max-sub-tokens",
        type=int,
        default=1_000_000,
        help="Maximum total tokens for sub-agent llm_query() calls (default: 1M)",
    )

    # Other parameters
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum RLM iterations (default: 50)",
    )
    parser.add_argument(
        "--image",
        default="python:3.11-slim",
        help="Docker image (default: python:3.11-slim)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable rich console output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    working_dir = os.path.abspath(args.working_dir)
    if not os.path.isdir(working_dir):
        print(f"Error: --working-dir does not exist: {working_dir}", file=sys.stderr)
        sys.exit(1)

    instructions_path = os.path.join(working_dir, INSTRUCTIONS_FILENAME)
    if not os.path.isfile(instructions_path):
        print(
            f"Error: {INSTRUCTIONS_FILENAME} not found in {working_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(instructions_path) as f:
        instructions = f.read()

    if not instructions.strip():
        print(f"Error: {INSTRUCTIONS_FILENAME} is empty", file=sys.stderr)
        sys.exit(1)

    # Setup logging in working directory
    log_dir = os.path.join(working_dir, "rlm-logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = RLMLogger(log_dir=log_dir, file_name="refactor")

    # Build RLM configuration
    backend_kwargs = build_backend_kwargs(args, is_sub=False)
    environment_kwargs = {"working_dir": working_dir, "image": args.image}

    # Validate that users aren't trying to use max_root_tokens / max_sub_tokens in backend_kwargs
    # (common mistake - these should be RLM constructor params, not backend params)
    forbidden = ["max_root_tokens", "max_sub_tokens"]
    for param in forbidden:
        if param in backend_kwargs:
            print(
                f"Error: {param} should be passed as --{param.replace('_', '-')}, "
                f"not as --backend-arg",
                file=sys.stderr,
            )
            sys.exit(1)

    other_backends = None
    other_backend_kwargs = None
    if args.sub_model:
        sub_kwargs = build_backend_kwargs(args, is_sub=True)
        # Validate sub backend_kwargs too
        for param in forbidden:
            if param in sub_kwargs:
                print(
                    f"Error: {param} should be passed as --{param.replace('_', '-')}, "
                    f"not as --sub-backend-arg",
                    file=sys.stderr,
                )
                sys.exit(1)
        other_backends = [args.sub_backend or args.backend]
        other_backend_kwargs = [sub_kwargs]

    rlm = RLM(
        backend=args.backend,
        backend_kwargs=backend_kwargs,
        environment="docker",
        environment_kwargs=environment_kwargs,
        custom_system_prompt=REFACTOR_SYSTEM_PROMPT,
        logger=logger,
        max_iterations=args.max_iterations,
        max_root_tokens=args.max_root_tokens,
        max_sub_tokens=args.max_sub_tokens,
        verbose=args.verbose,
        other_backends=other_backends,
        other_backend_kwargs=other_backend_kwargs,
    )

    # Use truncated instructions as root_prompt so the root LM sees the task summary
    root_prompt = instructions[:500] if len(instructions) > 500 else instructions
    result = rlm.completion(prompt=instructions, root_prompt=root_prompt)

    print("\n" + "=" * 60)
    print("REFACTORING COMPLETE")
    print("=" * 60)
    print(f"\n{result.response}")
    print(f"\nExecution time: {result.execution_time:.1f}s")
    print(f"Logs written to: {log_dir}")


if __name__ == "__main__":
    main()
