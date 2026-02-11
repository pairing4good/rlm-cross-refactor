"""
Programmatic usage of the RLM refactoring agent.

For most users, the CLI is simpler:
    # Basic usage
    rlm-refactor --working-dir ./my-repos --model gpt-4o --verbose

    # With inference parameters
    rlm-refactor --working-dir ./my-repos --model gpt-4o \\
      --max-tokens 8000 --temperature 0.7 --top-p 0.95

    # With sub-model configuration
    rlm-refactor --working-dir ./my-repos --model gpt-4o \\
      --sub-model gpt-4o-mini --sub-max-tokens 4000 --sub-temperature 0.3

    # With session-wide token budgets
    rlm-refactor --working-dir ./my-repos --model gpt-4o \\
      --max-root-tokens 500000 --max-sub-tokens 500000

    # With provider-specific parameters
    rlm-refactor --working-dir ./my-repos --model custom-model \\
      --backend-arg base_url=http://localhost:8080 \\
      --backend-arg api_key=custom-key

This example shows how to use the refactoring system prompt with
the RLM class directly for custom integration.

Setup:
    1. Ensure Docker is running
    2. Create a directory with git repos and an rlm-instructions.md file
    3. Run: python -m examples.refactor_example
"""

import os

from rlm.core.rlm import RLM
from rlm.logger import RLMLogger
from rlm.refactor.prompts import REFACTOR_SYSTEM_PROMPT


def main():
    working_dir = os.path.abspath("./example-refactor-workspace")

    # Read the task instructions
    instructions_path = os.path.join(working_dir, "rlm-instructions.md")
    if not os.path.isfile(instructions_path):
        print(f"Create {instructions_path} with your refactoring instructions first.")
        print("See examples/rlm-instructions-template.md for a template.")
        return

    with open(instructions_path) as f:
        instructions = f.read()

    # Setup logging inside the working directory
    log_dir = os.path.join(working_dir, "rlm-logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = RLMLogger(log_dir=log_dir, file_name="refactor")

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "gpt-4o"},
        environment="docker",
        environment_kwargs={
            "working_dir": working_dir,
            # Use a custom image with git pre-installed for faster startup:
            # "image": "my-rlm-refactor-image",
        },
        custom_system_prompt=REFACTOR_SYSTEM_PROMPT,
        logger=logger,
        max_iterations=50,
        verbose=True,
        # Optional: use a cheaper model for sub-LLM queries
        # other_backends=["openai"],
        # other_backend_kwargs=[{"model_name": "gpt-4o-mini"}],
    )

    result = rlm.completion(
        prompt=instructions,
        root_prompt=instructions[:500],
    )

    print(f"\nResult: {result.response}")
    print(f"Time: {result.execution_time:.1f}s")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
