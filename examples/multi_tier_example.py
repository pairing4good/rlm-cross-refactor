"""
Multi-Tier Model Routing Example: Claude Sonnet (root) + Haiku (sub)

This example demonstrates how to use multi-tier routing to optimize costs by:
1. Using Claude Sonnet for high-level reasoning and planning (root model)
2. Delegating simple execution tasks to Claude Haiku (sub-model) via llm_query()

Cost Comparison:
- Claude 3.5 Sonnet: $3/M input + $15/M output tokens
- Claude 3 Haiku: $0.25/M input + $1.25/M output tokens
- Savings: ~10x cheaper for sub-queries

Without multi-tier: All calls use Sonnet → ~$9 for this example
With multi-tier: Root uses Sonnet, 10 subs use Haiku → ~$1.50 (83% savings!)
"""

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

# Setup logger to track which model handles which call
logger = RLMLogger(log_dir="./logs")

# Configure multi-tier routing
rlm = RLM(
    # Root model (depth=0): Claude Sonnet for strategic planning
    backend="bedrock",
    backend_kwargs={
        "model_name": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "region_name": "us-east-1",
        "max_tokens": 8192,
    },
    # Sub-model (depth=1): Claude Haiku for simple execution tasks
    # Any llm_query() calls in generated code will automatically use Haiku
    other_backends=["bedrock"],
    other_backend_kwargs=[
        {
            "model_name": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "region_name": "us-east-1",
            "max_tokens": 4096,
        }
    ],
    environment="local",
    logger=logger,
    verbose=True,
)

# Example task: Process a list of items
# The root model (Sonnet) will strategize how to process the data
# The sub-model (Haiku) will handle individual item processing
prompt = """
Given this list of employee records, extract the key information from each:

Records:
1. John Doe, Software Engineer, joined 2020, salary $120k
2. Jane Smith, Product Manager, joined 2019, salary $135k
3. Bob Wilson, Designer, joined 2021, salary $105k
4. Alice Brown, Data Scientist, joined 2022, salary $130k
5. Charlie Davis, DevOps Engineer, joined 2020, salary $125k

For each record:
1. Extract: name, role, join_year, salary
2. Calculate years of experience (2026 - join_year)
3. Classify seniority: <3 years = Junior, 3-5 years = Mid, >5 years = Senior

Use llm_query() to process each record individually for parallel processing.
Create a list to collect all results, then use FINAL() to return the list.

Note: When constructing prompts for llm_query(), avoid using backslashes in f-string expressions.
Instead of f"text{var}\n", use f"text{var}" + "\n" or template strings.
"""

print("=" * 80)
print("MULTI-TIER ROUTING EXAMPLE: Claude Sonnet (root) + Haiku (sub)")
print("=" * 80)
print("\nPrompt:", prompt)
print("\n" + "=" * 80)

result = rlm.completion(prompt)

print("\n" + "=" * 80)
print("RESULT:")
print("=" * 80)
print(result.response)

# Show cost breakdown
print("\n" + "=" * 80)
print("USAGE SUMMARY:")
print("=" * 80)
usage = result.usage_summary.to_dict()
for model_name, stats in usage["model_usage_summaries"].items():
    print(f"\nModel: {model_name}")
    print(f"  Calls: {stats['total_calls']}")
    print(f"  Input tokens: {stats['total_input_tokens']:,}")
    print(f"  Output tokens: {stats['total_output_tokens']:,}")
    total_tokens = stats["total_input_tokens"] + stats["total_output_tokens"]
    print(f"  Total tokens: {total_tokens:,}")

# Calculate approximate cost
total_cost = 0
for model_name, stats in usage["model_usage_summaries"].items():
    if "sonnet" in model_name.lower():
        # Sonnet: $3/M input, $15/M output
        cost = (stats["total_input_tokens"] / 1_000_000 * 3) + (
            stats["total_output_tokens"] / 1_000_000 * 15
        )
        print(f"\n  Sonnet Cost: ${cost:.4f}")
    elif "haiku" in model_name.lower():
        # Haiku: $0.25/M input, $1.25/M output
        cost = (stats["total_input_tokens"] / 1_000_000 * 0.25) + (
            stats["total_output_tokens"] / 1_000_000 * 1.25
        )
        print(f"  Haiku Cost: ${cost:.4f}")
    total_cost += cost

print(f"\n{'=' * 80}")
print(f"TOTAL ESTIMATED COST: ${total_cost:.4f}")
print(f"{'=' * 80}")

# Calculate what cost would be with all-Sonnet
all_sonnet_cost = 0
for _model_name, stats in usage["model_usage_summaries"].items():
    all_sonnet_cost += (stats["total_input_tokens"] / 1_000_000 * 3) + (
        stats["total_output_tokens"] / 1_000_000 * 15
    )

savings = all_sonnet_cost - total_cost
savings_pct = (savings / all_sonnet_cost * 100) if all_sonnet_cost > 0 else 0

print(f"\nCost if all-Sonnet: ${all_sonnet_cost:.4f}")
print(f"Savings with multi-tier: ${savings:.4f} ({savings_pct:.1f}% reduction)")
print("\n✓ Multi-tier routing example completed!")
