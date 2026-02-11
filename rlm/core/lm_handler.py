"""
LMHandler - Routes LLM requests from the RLM process and environment subprocesses.

Uses a multi-threaded socket server. Protocol: 4-byte length prefix + JSON payload.
"""

import asyncio
import time
from socketserver import StreamRequestHandler, ThreadingTCPServer
from threading import Thread
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.comms_utils import LMRequest, LMResponse, socket_recv, socket_send
from rlm.core.types import RLMChatCompletion, UsageSummary


class LMRequestHandler(StreamRequestHandler):
    """Socket handler for LLM completion requests."""

    def handle(self):
        try:
            request_data = socket_recv(self.connection)
            if not isinstance(request_data, dict):
                response = LMResponse.error_response("Request must be a JSON object")
                self._safe_send(response)
                return

            request = LMRequest.from_dict(request_data)
            handler: LMHandler = self.server.lm_handler  # type: ignore

            if request.is_batched:
                # Batched request: process multiple prompts concurrently
                response = self._handle_batched(request, handler)
            elif request.prompt:
                # Single request: process one prompt
                response = self._handle_single(request, handler)
            else:
                response = LMResponse.error_response("Missing 'prompt' or 'prompts' in request.")

            self._safe_send(response)

        except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError):
            # Client disconnected - this is expected during parallel execution
            # when workers complete and close their sockets. Silently ignore.
            pass

        except Exception as e:
            # Try to send error response, but don't fail if socket is broken
            response = LMResponse.error_response(str(e))
            self._safe_send(response)

    def _safe_send(self, response: LMResponse) -> bool:
        """Send response, returning False if the socket is broken."""
        try:
            socket_send(self.connection, response.to_dict())
            return True
        except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError):
            # Client disconnected - silently ignore
            return False

    def _handle_single(self, request: LMRequest, handler: "LMHandler") -> LMResponse:
        """Handle a single prompt request."""
        assert request.prompt is not None, "Single request must have non-None prompt"

        # Check token limits before processing
        limit_check = handler._check_token_limits(request.depth)
        if limit_check is not None:
            return limit_check

        client = handler.get_client(request.model, request.depth)

        start_time = time.perf_counter()
        content = client.completion(request.prompt)
        end_time = time.perf_counter()

        model_usage = client.get_last_usage()
        root_model = request.model or client.model_name
        usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})
        return LMResponse.success_response(
            chat_completion=RLMChatCompletion(
                root_model=root_model,
                prompt=request.prompt,
                response=content,
                usage_summary=usage_summary,
                execution_time=end_time - start_time,
            )
        )

    def _handle_batched(self, request: LMRequest, handler: "LMHandler") -> LMResponse:
        """Handle a batched prompts request using async for concurrency."""
        assert request.prompts is not None, "Batched request must have non-None prompts"

        # Check token limits before processing
        limit_check = handler._check_token_limits(request.depth)
        if limit_check is not None:
            return limit_check

        client = handler.get_client(request.model, request.depth)

        start_time = time.perf_counter()

        async def run_all():
            assert request.prompts is not None  # Type narrowing for closure
            tasks = [client.acompletion(prompt) for prompt in request.prompts]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_all())
        end_time = time.perf_counter()

        total_time = end_time - start_time
        model_usage = client.get_last_usage()
        root_model = request.model or client.model_name
        usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})

        chat_completions = [
            RLMChatCompletion(
                root_model=root_model,
                prompt=prompt,
                response=content,
                usage_summary=usage_summary,
                execution_time=total_time / len(request.prompts),  # approximate per-prompt time
            )
            for prompt, content in zip(request.prompts, results, strict=True)
        ]

        return LMResponse.batched_success_response(chat_completions=chat_completions)


class ThreadingLMServer(ThreadingTCPServer):
    """Multi-threaded TCP server for LM requests."""

    daemon_threads = True
    allow_reuse_address = True


class LMHandler:
    """
    Handles all LM calls from the RLM main process and environment subprocesses.

    Uses a multi-threaded socket server for concurrent requests.
    Protocol: 4-byte big-endian length prefix + JSON payload.
    """

    def __init__(
        self,
        client: BaseLM,
        host: str = "127.0.0.1",
        port: int = 0,  # auto-assign available port
        other_backend_client: BaseLM | None = None,
        max_root_tokens: int | None = None,
        max_sub_tokens: int | None = None,
    ):
        self.default_client = client
        self.other_backend_client = other_backend_client
        self.clients: dict[str, BaseLM] = {}
        self.max_root_tokens = max_root_tokens
        self.max_sub_tokens = max_sub_tokens
        self.host = host
        self._server: ThreadingLMServer | None = None
        self._thread: Thread | None = None
        self._port = port

        # Track which depths use which clients for accurate token attribution
        self._depth_client_mapping: dict[int, set[str]] = {0: set(), 1: set()}

        self.register_client(client.model_name, client)

    def register_client(self, model_name: str, client: BaseLM) -> None:
        """Register a client for a specific model name."""
        self.clients[model_name] = client

    def get_client(self, model: str | None = None, depth: int = 0) -> BaseLM:
        """Get client by model name or depth, or return default.

        Routing logic:
        - depth=0: use default_client (main backend)
        - depth=1: use other_backend_client if it exists, otherwise default_client
        - If model is specified and exists in clients, use that (overrides depth routing)
        """
        if model and model in self.clients:
            return self.clients[model]

        # Route based on depth
        if depth == 1 and self.other_backend_client is not None:
            return self.other_backend_client

        return self.default_client

    @property
    def port(self) -> int:
        """Get the actual port (useful when auto-assigned)."""
        if self._server:
            return self._server.server_address[1]
        return self._port

    @property
    def address(self) -> tuple[str, int]:
        """Get (host, port) tuple for connecting."""
        return (self.host, self.port)

    def start(self) -> tuple[str, int]:
        """Start the socket server in a background thread. Returns (host, port)."""
        if self._server is not None:
            return self.address

        self._server = ThreadingLMServer((self.host, self._port), LMRequestHandler)
        self._server.lm_handler = self  # type: ignore

        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        return self.address

    def stop(self):
        """Stop the socket server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None

    def completion(
        self, prompt: str | dict[str, Any] | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """Direct completion call (for main process use)."""
        return self.get_client(model).completion(prompt)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def get_usage_summary(self) -> UsageSummary:
        """Get the usage summary for all clients, merged into a single dict."""
        merged = {}
        # Include default client
        default_summary = self.default_client.get_usage_summary()
        merged.update(default_summary.model_usage_summaries)
        # Include other backend client if it exists
        if self.other_backend_client is not None:
            other_summary = self.other_backend_client.get_usage_summary()
            merged.update(other_summary.model_usage_summaries)
        # Include all registered clients
        for client in self.clients.values():
            client_summary = client.get_usage_summary()
            merged.update(client_summary.model_usage_summaries)
        return UsageSummary(model_usage_summaries=merged)

    def get_depth_specific_usage(self) -> tuple[int, int]:
        """Get separate token counts for root (depth=0) and sub-agent (depth=1) calls.

        Returns:
            tuple[int, int]: (root_tokens, sub_tokens) where each is total input + output tokens
        """
        root_tokens = 0
        sub_tokens = 0

        # Default client is always used for root calls (depth=0)
        default_usage = self.default_client.get_usage_summary()
        for model_usage in default_usage.model_usage_summaries.values():
            root_tokens += model_usage.total_input_tokens + model_usage.total_output_tokens

        # Other backend client is used for sub-agent calls (depth=1)
        if self.other_backend_client is not None:
            other_usage = self.other_backend_client.get_usage_summary()
            for model_usage in other_usage.model_usage_summaries.values():
                sub_tokens += model_usage.total_input_tokens + model_usage.total_output_tokens

        return root_tokens, sub_tokens

    def _check_token_limits(self, depth: int) -> LMResponse | None:
        """Check if token limits have been exceeded for the given depth.

        Uses conservative estimation: blocks call if current usage + buffer would exceed limit.
        This prevents the first call from happening when limits are too low.

        Args:
            depth: The depth of the request (0=root, 1=sub-agent)

        Returns:
            LMResponse with error if limit exceeded, None if within limits
        """
        root_tokens, sub_tokens = self.get_depth_specific_usage()

        # Conservative buffer: typical minimum tokens per LM call
        # (small prompt + small response = ~50-100 tokens minimum)
        CONSERVATIVE_BUFFER = 50

        # Check root limit for depth=0 calls
        if depth == 0 and self.max_root_tokens is not None:
            # Block if current usage + buffer would exceed limit
            if root_tokens + CONSERVATIVE_BUFFER >= self.max_root_tokens:
                return LMResponse.error_response(
                    f"TOKEN_LIMIT_EXCEEDED:root:{root_tokens}:{self.max_root_tokens}"
                )

        # Check sub-agent limit for depth=1 calls
        if depth == 1 and self.max_sub_tokens is not None:
            # Block if current usage + buffer would exceed limit
            if sub_tokens + CONSERVATIVE_BUFFER >= self.max_sub_tokens:
                return LMResponse.error_response(
                    f"TOKEN_LIMIT_EXCEEDED:sub_agent:{sub_tokens}:{self.max_sub_tokens}"
                )

        return None
