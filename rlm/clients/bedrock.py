from collections import defaultdict
from typing import Any

import boto3

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class BedrockClient(BaseLM):
    """
    LM Client for running models with AWS Bedrock.

    Supports all Bedrock models via the Converse API including:
    - Anthropic Claude (us.anthropic.claude-3-5-sonnet-20241022-v2:0, etc.)
    - Meta Llama (meta.llama3-70b-instruct-v1:0, etc.)
    - Mistral AI (mistral.mistral-large-2402-v1:0, etc.)
    - Amazon Titan (amazon.titan-text-express-v1, etc.)

    Example:
        client = BedrockClient(
            model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-1",
            aws_access_key_id="...",  # Optional if using IAM role/AWS CLI config
            aws_secret_access_key="...",  # Optional
        )
    """

    def __init__(
        self,
        model_name: str,
        region_name: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.999,
        **kwargs,
    ):
        """
        Initialize AWS Bedrock client.

        Args:
            model_name: Bedrock model ID (e.g., "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
            region_name: AWS region where Bedrock is available (default: us-east-1)
            aws_access_key_id: AWS access key (optional if using IAM role or AWS CLI config)
            aws_secret_access_key: AWS secret key (optional)
            aws_session_token: AWS session token for temporary credentials (optional)
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Sampling temperature 0.0-1.0 (default: 1.0)
            top_p: Nucleus sampling threshold (default: 0.999)
        """
        super().__init__(model_name=model_name, **kwargs)

        # Build boto3 session kwargs
        session_kwargs = {"region_name": region_name}
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

        # Create Bedrock Runtime client
        self.client = boto3.client("bedrock-runtime", **session_kwargs)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

        # Last call tracking
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def completion(
        self, prompt: str | dict[str, Any] | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """Synchronous completion using Bedrock Converse API."""
        messages, system = self._prepare_messages(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Bedrock client.")

        # Build request kwargs
        request_kwargs = {
            "modelId": model,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            },
        }

        # Add system prompt if present
        if system:
            request_kwargs["system"] = [{"text": system}]

        # Make API call
        response = self.client.converse(**request_kwargs)

        # Track usage
        self._track_usage(response, model)

        # Extract response text
        return response["output"]["message"]["content"][0]["text"]

    async def acompletion(
        self, prompt: str | dict[str, Any] | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """
        Async completion using Bedrock Converse API.

        Note: boto3 doesn't have native async support, so this uses sync implementation.
        For production async workloads, consider using aioboto3.
        """
        # boto3 is synchronous - we just call the sync method
        # For true async, would need aioboto3 (not adding extra dependency)
        return self.completion(prompt, model)

    def _prepare_messages(
        self, prompt: str | dict[str, Any] | list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Prepare messages for Bedrock Converse API.

        Converts from RLM message format to Bedrock format.
        Extracts system messages separately (Bedrock handles them differently).
        """
        system = None

        if isinstance(prompt, str):
            # Simple string prompt
            messages = [{"role": "user", "content": [{"text": prompt}]}]
        elif isinstance(prompt, dict):
            # Single message dict
            role = prompt.get("role")
            content = prompt.get("content", "")
            if role == "system":
                system = content
                messages = []
            elif role in ("user", "assistant"):
                messages = [{"role": role, "content": [{"text": content}]}]
            else:
                raise ValueError(f"Unsupported role: {role}")
        elif isinstance(prompt, list):
            # Message list format
            messages = []
            for msg in prompt:
                role = msg.get("role")
                content = msg.get("content", "")

                if role == "system":
                    # Extract system message (handled separately in Bedrock)
                    system = content
                elif role in ("user", "assistant"):
                    # Convert to Bedrock message format
                    messages.append(
                        {
                            "role": role,
                            "content": [{"text": content}],
                        }
                    )
                else:
                    raise ValueError(f"Unsupported role: {role}")
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        return messages, system

    def _track_usage(self, response: dict[str, Any], model: str):
        """Track token usage from Bedrock response."""
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += input_tokens
        self.model_output_tokens[model] += output_tokens
        self.model_total_tokens[model] += input_tokens + output_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = input_tokens
        self.last_completion_tokens = output_tokens

    def get_usage_summary(self) -> UsageSummary:
        """Get aggregated usage across all models."""
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        """Get usage for the most recent call."""
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
