from collections import defaultdict
from typing import Any, cast

from portkey_ai import AsyncPortkey, Portkey
from portkey_ai.api_resources.types.chat_complete_type import ChatCompletions

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class PortkeyClient(BaseLM):
    """
    LM Client for running models with the Portkey API.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str | None = None,
        base_url: str | None = "https://api.portkey.ai/v1",
        **kwargs,
    ):
        super().__init__(model_name=model_name or "unknown", **kwargs)
        self.client = Portkey(api_key=api_key, base_url=base_url)
        self.async_client = AsyncPortkey(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

    def completion(
        self, prompt: str | dict[str, Any] | list[dict[str, Any]], model: str | None = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            messages = [prompt]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Portkey client.")

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
        )
        # Cast to ensure ty understands this is not streaming
        response = cast(ChatCompletions, response)
        self._track_cost(response, model)
        content = response.choices[0].message.content if response.choices else None
        return content or ""

    async def acompletion(
        self, prompt: str | dict[str, Any] | list[dict[str, Any]], model: str | None = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            messages = [prompt]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Portkey client.")

        response = await self.async_client.chat.completions.create(
            model=model, messages=messages, stream=False
        )
        # Cast to ensure ty understands this is not streaming
        response = cast(ChatCompletions, response)
        self._track_cost(response, model)
        content = response.choices[0].message.content if response.choices else None
        return content or ""

    def _track_cost(self, response: ChatCompletions, model: str):
        if not response.usage:
            return

        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += response.usage.prompt_tokens or 0
        self.model_output_tokens[model] += response.usage.completion_tokens or 0
        self.model_total_tokens[model] += response.usage.total_tokens or 0

        # Track last call for handler to read
        self.last_prompt_tokens = response.usage.prompt_tokens or 0
        self.last_completion_tokens = response.usage.completion_tokens or 0

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=getattr(self, "last_prompt_tokens", 0),
            total_output_tokens=getattr(self, "last_completion_tokens", 0),
        )
