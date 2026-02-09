import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

# Example 1: Using IAM role or AWS CLI configured credentials (recommended)
print("=== Example 1: Using IAM role or AWS CLI credentials ===")

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="bedrock",
    backend_kwargs={
        "model_name": "us.us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "region_name": "us-east-1",
        "max_tokens": 8192,
    },
    environment="docker",
    environment_kwargs={},
    max_depth=1,
    logger=logger,
    verbose=True,
)

result = rlm.completion("Calculate the first 5 powers of two, each on a newline.")
print(f"\nResult: {result.response}\n")

print("✓ All Bedrock examples completed successfully!")
