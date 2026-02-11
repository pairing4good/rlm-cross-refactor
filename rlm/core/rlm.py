import time
from contextlib import contextmanager
from typing import Any, cast

from rlm.clients import BaseLM, get_client
from rlm.core.lm_handler import LMHandler
from rlm.core.types import (
    ClientBackend,
    CodeBlock,
    EnvironmentType,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
    UsageSummary,
)
from rlm.environments import BaseEnv, SupportsPersistence, get_environment
from rlm.logger import RLMLogger, VerbosePrinter
from rlm.utils.parsing import (
    find_code_blocks,
    find_final_answer,
    format_iteration,
)
from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)
from rlm.utils.rlm_utils import filter_sensitive_keys


class RLM:
    """
    Recursive Language Model class that the user instantiates and runs on their tasks.

    Each completion() call spawns its own environment and LM handler, which are
    cleaned up when the call completes.
    """

    def __init__(
        self,
        backend: ClientBackend = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        environment: EnvironmentType = "local",
        environment_kwargs: dict[str, Any] | None = None,
        depth: int = 0,
        max_depth: int = 1,
        max_iterations: int = 30,
        max_root_tokens: int | None = 1_000_000,
        max_sub_tokens: int | None = 1_000_000,
        custom_system_prompt: str | None = None,
        other_backends: list[ClientBackend] | None = None,
        other_backend_kwargs: list[dict[str, Any]] | None = None,
        logger: RLMLogger | None = None,
        verbose: bool = False,
        persistent: bool = False,
    ):
        """
        Args:
            backend: The backend to use for the RLM.
            backend_kwargs: The kwargs to pass to the backend. Should NOT include token limits -
                use max_root_tokens/max_sub_tokens constructor params for session-wide limits.
            environment: The environment to use for the RLM.
            environment_kwargs: The kwargs to pass to the environment.
            depth: The current depth of the RLM (0-indexed).
            max_depth: The maximum depth of the RLM. Currently, only depth 1 is supported.
            max_iterations: The maximum number of iterations of the RLM.
            max_root_tokens: The maximum total tokens (input + output) allowed for root agent calls (depth=0).
                DEFAULT: 1,000,000 tokens to prevent unexpectedly large API bills.
                COST ESTIMATE: 1M tokens costs ~$1-45 depending on your model:
                  - GPT-3.5 Turbo: ~$1  |  GPT-4o: ~$6  |  GPT-4 Turbo: ~$20  |  Claude Opus: ~$45
                Set to None for unlimited tokens (use with caution - can result in expensive bills).
                Set to a smaller value (e.g., 100_000) for tighter budget control.
            max_sub_tokens: The maximum total tokens (input + output) allowed for sub-agent calls (depth=1).
                This limits the cumulative tokens used by llm_query() calls within generated code.
                DEFAULT: 1,000,000 tokens. Set to None for unlimited sub-agent tokens.
            custom_system_prompt: The custom system prompt to use for the RLM.
            other_backends: A list of other client backends that the environments can use to make sub-calls.
            other_backend_kwargs: The kwargs to pass to the other client backends (ordered to match other_backends).
                Should NOT include token limits - use max_sub_tokens constructor param for session-wide limits.
            logger: The logger to use for the RLM.
            verbose: Whether to print verbose output in rich to console.
            persistent: If True, reuse the environment across completion() calls for multi-turn conversations.
        """
        # Validate backend_kwargs for problematic token limit settings
        if backend_kwargs:
            self._validate_backend_kwargs(backend_kwargs, "backend_kwargs (root)")
        if other_backend_kwargs:
            for i, kwargs in enumerate(other_backend_kwargs):
                self._validate_backend_kwargs(kwargs, f"other_backend_kwargs[{i}] (sub)")

        # Store config for spawning per-completion
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.environment_type = environment
        self.environment_kwargs = (
            environment_kwargs.copy() if environment_kwargs is not None else {}
        )
        # Validate other_backends: currently only support one additional backend
        if other_backends is not None:
            if len(other_backends) != 1:
                raise ValueError(
                    "We currently only support one additional backend for the recursive sub-calls! "
                    "This model will be the model used for recursive sub-calls, but this will change in the future"
                )

        self.other_backends = other_backends
        self.other_backend_kwargs = other_backend_kwargs

        self.depth = depth
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.max_root_tokens = max_root_tokens
        self.max_sub_tokens = max_sub_tokens
        self.system_prompt = custom_system_prompt if custom_system_prompt else RLM_SYSTEM_PROMPT
        self.logger = logger
        self.verbose = VerbosePrinter(enabled=verbose)

        # Persistence support
        self.persistent = persistent
        self._persistent_env: SupportsPersistence | None = None

        # Validate persistence support at initialization
        if self.persistent:
            self._validate_persistent_environment_support()

        # Log metadata if logger is provided
        if self.logger or verbose:
            metadata = RLMMetadata(
                root_model=backend_kwargs.get("model_name", "unknown")
                if backend_kwargs
                else "unknown",
                max_depth=max_depth,
                max_iterations=max_iterations,
                max_root_tokens=max_root_tokens,
                max_sub_tokens=max_sub_tokens,
                backend=backend,
                backend_kwargs=filter_sensitive_keys(backend_kwargs) if backend_kwargs else {},
                environment_type=environment,
                environment_kwargs=filter_sensitive_keys(environment_kwargs)
                if environment_kwargs
                else {},
                other_backends=cast(list[str] | None, other_backends),
            )
            if self.logger:
                self.logger.log_metadata(metadata)
            self.verbose.print_metadata(metadata)

    @contextmanager
    def _spawn_completion_context(self, prompt: str | dict[str, Any]):
        """
        Spawn an LM handler and environment for a single completion call.

        When persistent=True, the environment is reused across calls.
        When persistent=False (default), creates fresh environment each call.
        """
        # Create client and wrap in handler
        client: BaseLM = get_client(self.backend, self.backend_kwargs or {})

        # Create other_backend_client if provided (for depth=1 routing)
        other_backend_client: BaseLM | None = None
        if self.other_backends and self.other_backend_kwargs:
            other_backend_client = get_client(
                self.other_backends[0], self.other_backend_kwargs[0] or {}
            )

        lm_handler = LMHandler(
            client,
            other_backend_client=other_backend_client,
            max_root_tokens=self.max_root_tokens,
            max_sub_tokens=self.max_sub_tokens,
        )

        # Register other clients to be available as sub-call options (by model name)
        if self.other_backends and self.other_backend_kwargs:
            for backend, kwargs in zip(self.other_backends, self.other_backend_kwargs, strict=True):
                other_client: BaseLM = get_client(backend, kwargs)
                lm_handler.register_client(other_client.model_name, other_client)

        lm_handler.start()

        # Environment: reuse if persistent, otherwise create fresh
        if self.persistent and self._persistent_env is not None:
            environment = self._persistent_env
            # Defensive check: ensure environment supports persistence methods
            if not self._env_supports_persistence(environment):
                raise RuntimeError(
                    f"Persistent environment of type '{type(environment).__name__}' does not "
                    f"implement required methods (update_handler_address, add_context, get_context_count). "
                    f"This should have been caught at initialization."
                )
            environment.update_handler_address((lm_handler.host, lm_handler.port))
            environment.add_context(prompt)
        else:
            env_kwargs = self.environment_kwargs.copy()
            env_kwargs["lm_handler_address"] = (lm_handler.host, lm_handler.port)
            env_kwargs["context_payload"] = prompt
            env_kwargs["depth"] = self.depth + 1  # Environment depth is RLM depth + 1
            environment: BaseEnv = get_environment(self.environment_type, env_kwargs)

            if self.persistent:
                self._persistent_env = environment

        try:
            yield lm_handler, environment
        finally:
            lm_handler.stop()
            if not self.persistent:
                cleanup_fn = getattr(environment, "cleanup", None)
                if cleanup_fn is not None:
                    cleanup_fn()

    def _setup_prompt(self, prompt: str | dict[str, Any]) -> list[dict[str, Any]]:
        """
        Setup the system prompt for the RLM. Also include metadata about the prompt and build
        up the initial message history.
        """
        metadata = QueryMetadata(prompt)
        message_history = build_rlm_system_prompt(
            system_prompt=self.system_prompt, query_metadata=metadata
        )

        return message_history

    def completion(
        self, prompt: str | dict[str, Any], root_prompt: str | None = None
    ) -> RLMChatCompletion:
        """
        Recursive Language Model completion call. This is the main entry point for querying an RLM, and
        can replace a regular LM completion call.

        Spawns its own environment and LM handler for the duration of this call.

        Args:
            prompt: A single string or dictionary of messages to pass as context to the model.
            root_prompt: We allow the RLM's root LM to see a (small) prompt that the user specifies. A common example of this
            is if the user is asking the RLM to answer a question, we can pass the question as the root prompt.
        Returns:
            A final answer as a string.
        """
        time_start = time.perf_counter()

        # If we're at max depth, the RLM is an LM, so we fallback to the regular LM.
        if self.depth >= self.max_depth:
            return self._fallback_answer(prompt)

        with self._spawn_completion_context(prompt) as (lm_handler, environment):
            message_history = self._setup_prompt(prompt)

            # Pre-flight check: Validate limits aren't absurdly low
            # A single LM call typically uses 50-200+ tokens minimum
            MIN_VIABLE_TOKENS = 50
            if self.max_root_tokens is not None and self.max_root_tokens < MIN_VIABLE_TOKENS:
                # Immediately fail - limit too low for even one call
                return self._token_limit_exceeded_answer(
                    0, 0, time_start, lm_handler, limit_type="root"
                )
            if self.max_sub_tokens is not None and self.max_sub_tokens < MIN_VIABLE_TOKENS:
                # Immediately fail - limit too low for even one call
                return self._token_limit_exceeded_answer(
                    0, 0, time_start, lm_handler, limit_type="sub_agent"
                )

            for i in range(self.max_iterations):
                # Check token limits before starting next iteration
                if self.max_root_tokens is not None or self.max_sub_tokens is not None:
                    root_tokens, sub_tokens = lm_handler.get_depth_specific_usage()

                    # Check root token limit
                    if self.max_root_tokens is not None and root_tokens >= self.max_root_tokens:
                        if self.persistent and isinstance(environment, SupportsPersistence):
                            environment.add_history(message_history)
                        return self._token_limit_exceeded_answer(
                            i, root_tokens, time_start, lm_handler, limit_type="root"
                        )

                    # Check sub-agent token limit
                    if self.max_sub_tokens is not None and sub_tokens >= self.max_sub_tokens:
                        if self.persistent and isinstance(environment, SupportsPersistence):
                            environment.add_history(message_history)
                        return self._token_limit_exceeded_answer(
                            i, sub_tokens, time_start, lm_handler, limit_type="sub_agent"
                        )

                # Current prompt = message history + additional prompt suffix
                context_count = (
                    environment.get_context_count()
                    if isinstance(environment, SupportsPersistence)
                    else 1
                )
                history_count = (
                    environment.get_history_count()
                    if isinstance(environment, SupportsPersistence)
                    else 0
                )
                current_prompt = message_history + [
                    build_user_prompt(root_prompt, i, context_count, history_count)
                ]

                iteration: RLMIteration = self._completion_turn(
                    prompt=current_prompt,
                    lm_handler=lm_handler,
                    environment=environment,
                )

                # Check if any code block hit a token limit during execution
                for code_block in iteration.code_blocks:
                    if code_block.result.token_limit_exceeded:
                        # Token limit hit during sub-agent call - terminate immediately
                        if self.persistent and isinstance(environment, SupportsPersistence):
                            environment.add_history(message_history)

                        limit_details = code_block.result.limit_details or {}
                        tokens_used = limit_details.get("tokens_used", 0)
                        return self._token_limit_exceeded_answer(
                            i,
                            tokens_used,
                            time_start,
                            lm_handler,
                            limit_type=code_block.result.limit_type or "sub_agent",
                        )

                # Check if RLM is done and has a final answer.
                final_answer = find_final_answer(iteration.response, environment=environment)
                iteration.final_answer = final_answer

                # If logger is used, log the iteration.
                if self.logger:
                    self.logger.log(iteration)

                # Verbose output for this iteration
                self.verbose.print_iteration(iteration, i + 1)

                if final_answer is not None:
                    time_end = time.perf_counter()
                    usage = lm_handler.get_usage_summary()
                    self.verbose.print_final_answer(final_answer)
                    self.verbose.print_summary(i + 1, time_end - time_start, usage.to_dict())

                    # Store message history in persistent environment
                    if self.persistent and isinstance(environment, SupportsPersistence):
                        environment.add_history(message_history)

                    return RLMChatCompletion(
                        root_model=self.backend_kwargs.get("model_name", "unknown")
                        if self.backend_kwargs
                        else "unknown",
                        prompt=prompt,
                        response=final_answer,
                        usage_summary=usage,
                        execution_time=time_end - time_start,
                    )

                # Format the iteration for the next prompt.
                new_messages = format_iteration(iteration)

                # Update message history with the new messages.
                message_history.extend(new_messages)

            # Default behavior: we run out of iterations, provide one final answer
            time_end = time.perf_counter()
            final_answer = self._default_answer(message_history, lm_handler)
            usage = lm_handler.get_usage_summary()
            self.verbose.print_final_answer(final_answer)
            self.verbose.print_summary(self.max_iterations, time_end - time_start, usage.to_dict())

            # Store message history in persistent environment
            if self.persistent and isinstance(environment, SupportsPersistence):
                environment.add_history(message_history)

            return RLMChatCompletion(
                root_model=self.backend_kwargs.get("model_name", "unknown")
                if self.backend_kwargs
                else "unknown",
                prompt=prompt,
                response=final_answer,
                usage_summary=usage,
                execution_time=time_end - time_start,
            )

    def _completion_turn(
        self,
        prompt: str | dict[str, Any] | list[dict[str, Any]],
        lm_handler: LMHandler,
        environment: BaseEnv,
    ) -> RLMIteration:
        """
        Perform a single iteration of the RLM, including prompting the model
        and code execution + tool execution.
        """
        iter_start = time.perf_counter()
        response = lm_handler.completion(prompt)
        code_block_strs = find_code_blocks(response)
        code_blocks = []

        for code_block_str in code_block_strs:
            code_result: REPLResult = environment.execute_code(code_block_str)
            code_blocks.append(CodeBlock(code=code_block_str, result=code_result))

        iteration_time = time.perf_counter() - iter_start
        return RLMIteration(
            prompt=prompt,
            response=response,
            code_blocks=code_blocks,
            iteration_time=iteration_time,
        )

    def _default_answer(self, message_history: list[dict[str, Any]], lm_handler: LMHandler) -> str:
        """
        Default behavior if the RLM runs out of iterations and does not find a final answer.
        It will take the message history, and try to generate a final answer from it.
        """
        current_prompt = message_history + [
            {
                "role": "assistant",
                "content": "Please provide a final answer to the user's question based on the information provided.",
            }
        ]
        response = lm_handler.completion(current_prompt)

        if self.logger:
            self.logger.log(
                RLMIteration(
                    prompt=current_prompt,
                    response=response,
                    final_answer=response,
                    code_blocks=[],
                )
            )

        return response

    def _fallback_answer(self, message: str | dict[str, Any]) -> RLMChatCompletion:
        """
        Fallback behavior if the RLM is actually at max depth, and should be treated as an LM.
        """
        client: BaseLM = get_client(self.backend, self.backend_kwargs or {})
        start_time = time.perf_counter()
        response = client.completion(message)
        end_time = time.perf_counter()

        usage = client.get_last_usage()
        return RLMChatCompletion(
            root_model=self.backend_kwargs.get("model_name", client.model_name)
            if self.backend_kwargs
            else client.model_name,
            prompt=message,
            response=response,
            usage_summary=UsageSummary(model_usage_summaries={client.model_name: usage}),
            execution_time=end_time - start_time,
        )

    def _calculate_total_tokens(self, usage: UsageSummary) -> int:
        """Calculate total tokens (input + output) from usage summary."""
        total = 0
        for model_usage in usage.model_usage_summaries.values():
            total += model_usage.total_input_tokens + model_usage.total_output_tokens
        return total

    def _token_limit_exceeded_answer(
        self,
        iteration: int,
        tokens_used: int,
        time_start: float,
        lm_handler: LMHandler,
        limit_type: str = "root",
    ) -> RLMChatCompletion:
        """
        Handle early termination when token limit is exceeded.
        Returns a completion indicating the session ended due to token limit.

        Args:
            limit_type: Either "root" or "sub_agent" to indicate which limit was exceeded.
        """
        # Type narrowing: determine which limit was exceeded
        max_limit_value = self.max_root_tokens if limit_type == "root" else self.max_sub_tokens
        assert max_limit_value is not None

        time_end = time.perf_counter()
        usage = lm_handler.get_usage_summary()

        # Log the limit hit event
        limit_name = "root_token_limit" if limit_type == "root" else "sub_agent_token_limit"
        if self.logger:
            self.logger.log_limit_hit(
                limit_type=limit_name,
                max_value=max_limit_value,
                current_value=tokens_used,
                iteration=iteration,
            )

        # Print verbose output
        self.verbose.print_token_limit_hit(
            token_limit=max_limit_value,
            tokens_used=tokens_used,
            iteration=iteration,
            limit_type=limit_type,
        )
        self.verbose.print_summary(
            iteration, time_end - time_start, usage.to_dict(), stopped_reason="token_limit"
        )

        agent_type = "Root agent" if limit_type == "root" else "Sub-agent"
        response_message = (
            f"Session ended: {agent_type} token limit exceeded. "
            f"Used {tokens_used:,} tokens (limit: {max_limit_value:,}). "
            f"Completed {iteration} iteration(s) before reaching limit."
        )

        return RLMChatCompletion(
            root_model=self.backend_kwargs.get("model_name", "unknown")
            if self.backend_kwargs
            else "unknown",
            prompt="",
            response=response_message,
            usage_summary=usage,
            execution_time=time_end - time_start,
        )

    def _validate_persistent_environment_support(self) -> None:
        """
        Validate that the configured environment type supports persistent mode.

        Persistent mode requires environments to implement:
        - update_handler_address(address): Update LM handler address between calls
        - add_context(payload, index): Add new context for multi-turn conversations
        - get_context_count(): Return the number of loaded contexts

        Currently only 'local' (LocalREPL) supports these methods.

        Raises:
            ValueError: If the environment type does not support persistent mode.
        """
        # Known environments that support persistence
        persistent_supported_environments = {"local"}

        if self.environment_type not in persistent_supported_environments:
            raise ValueError(
                f"persistent=True is not supported for environment type '{self.environment_type}'. "
                f"Persistent mode requires environments that implement update_handler_address(), "
                f"add_context(), and get_context_count(). "
                f"Supported environments: {sorted(persistent_supported_environments)}"
            )

    @staticmethod
    def _env_supports_persistence(env: BaseEnv | SupportsPersistence) -> bool:
        """Check if an environment instance supports persistent mode methods."""
        return isinstance(env, SupportsPersistence)

    @staticmethod
    def _validate_backend_kwargs(kwargs: dict[str, Any], context: str) -> None:
        """Validate backend_kwargs to catch common mistakes that cause confusing behavior.

        Args:
            kwargs: The backend_kwargs dictionary to validate
            context: Description of where these kwargs came from (for error messages)

        Raises:
            ValueError: If deprecated or problematic parameters are found
        """
        # Check for deprecated max_root_tokens/max_sub_tokens in backend_kwargs
        if "max_root_tokens" in kwargs:
            raise ValueError(
                f"Found 'max_root_tokens' in {context}. "
                f"Token limits should be set via RLM constructor parameters (max_root_tokens=...), "
                f"NOT in backend_kwargs. Remove 'max_root_tokens' from {context}. "
                f"\n\nExample:\n"
                f"  RLM(\n"
                f"      backend_kwargs={{'model_name': 'gpt-4o'}},  # No token limit here\n"
                f"      max_root_tokens=100000,  # Session-wide limit (correct place)\n"
                f"  )"
            )

        if "max_sub_tokens" in kwargs:
            raise ValueError(
                f"Found 'max_sub_tokens' in {context}. "
                f"Token limits should be set via RLM constructor parameters (max_sub_tokens=...), "
                f"NOT in backend_kwargs. Remove 'max_sub_tokens' from {context}. "
                f"\n\nExample:\n"
                f"  RLM(\n"
                f"      other_backend_kwargs=[{{'model_name': 'gpt-3.5-turbo'}}],  # No token limit here\n"
                f"      max_sub_tokens=50000,  # Session-wide limit (correct place)\n"
                f"  )"
            )

        # Check for unreasonably low max_tokens (per-response limit that clients understand)
        if "max_tokens" in kwargs:
            max_tokens = kwargs["max_tokens"]
            if isinstance(max_tokens, int) and max_tokens < 100:
                raise ValueError(
                    f"Found 'max_tokens'={max_tokens} in {context}. "
                    f"This is unreasonably low and will cause the model to generate incomplete responses. "
                    f"\n\nIf you want to limit total SESSION tokens, use RLM constructor parameters instead:\n"
                    f"  RLM(\n"
                    f"      backend_kwargs={{'model_name': 'gpt-4o'}},  # Remove max_tokens\n"
                    f"      max_root_tokens=100000,  # Session-wide limit (correct place)\n"
                    f"  )\n\n"
                    f"If you really need a low per-response limit for the client, use at least 100 tokens."
                )

    def close(self) -> None:
        """Clean up persistent environment. Call when done with multi-turn conversations."""
        if self._persistent_env is not None:
            cleanup_fn = getattr(self._persistent_env, "cleanup", None)
            if cleanup_fn is not None:
                cleanup_fn()
            self._persistent_env = None

    def __enter__(self) -> "RLM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
