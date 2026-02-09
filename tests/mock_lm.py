from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class MockLM(BaseLM):
    """Simple mock LM that echoes prompts."""

    def __init__(self):
        super().__init__(model_name="mock-model")

    def completion(self, prompt):
        return f"Mock response to: {prompt[:50]}"

    async def acompletion(self, prompt):
        return self.completion(prompt)

    def get_usage_summary(self):
        return UsageSummary(
            model_usage_summaries={
                "mock-model": ModelUsageSummary(
                    total_calls=1, total_input_tokens=10, total_output_tokens=10
                )
            }
        )

    def get_last_usage(self):
        return self.get_usage_summary()


class MockLMWithTokenTracking(BaseLM):
    """Mock LM that tracks cumulative token usage across calls."""

    def __init__(self, tokens_per_call: int = 5000, model_name: str = "mock-model"):
        """
        Args:
            tokens_per_call: Number of tokens (input + output combined) consumed per completion call.
            model_name: Name of the model for tracking.
        """
        super().__init__(model_name=model_name)
        self.tokens_per_call = tokens_per_call
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def completion(self, prompt):
        """Simulate a completion call and track token usage."""
        self.call_count += 1
        # Split tokens evenly between input and output
        input_tokens = self.tokens_per_call // 2
        output_tokens = self.tokens_per_call - input_tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        return f"Mock response {self.call_count}"

    async def acompletion(self, prompt):
        return self.completion(prompt)

    def get_usage_summary(self):
        """Return cumulative usage summary."""
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=self.call_count,
                    total_input_tokens=self.total_input_tokens,
                    total_output_tokens=self.total_output_tokens,
                )
            }
        )

    def get_last_usage(self):
        """Return usage for the last call only."""
        input_tokens = self.tokens_per_call // 2
        output_tokens = self.tokens_per_call - input_tokens
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=input_tokens,
                    total_output_tokens=output_tokens,
                )
            }
        )
