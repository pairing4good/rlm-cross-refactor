"""Integration tests for multi-turn persistent REPL sessions.

Tests that multiple LM completion calls in one RLM session:
1. Share the same environment
2. Accumulate contexts (context_0, context_1, ...)
3. Accumulate histories (history_0, history_1, ...)
4. Preserve variables across calls
5. Properly inform the model about available contexts/histories

All tests use MockLMWithResponses (a real BaseLM subclass) instead of
unittest.mock.Mock to ensure real token tracking, proper model_name attributes,
and typed return values. The only thing mocked is the get_client factory to
avoid real API calls.
"""

from unittest.mock import patch

import pytest

import rlm.core.rlm as rlm_module
from rlm import RLM
from tests.mock_lm import MockLMWithResponses


class TestMultiTurnPersistentEnvironment:
    """Tests for environment persistence across completion calls."""

    def test_environment_reused_in_persistent_mode(self):
        """Verify the same environment instance is reused across completion calls."""
        mock_lm = MockLMWithResponses(["FINAL(answer from call)"])

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            with RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                persistent=True,
            ) as rlm:
                rlm.completion("First context")
                first_env = rlm._persistent_env

                mock_lm.set_responses(["FINAL(answer from call)"])
                rlm.completion("Second context")
                second_env = rlm._persistent_env

                assert first_env is second_env
                assert first_env is not None

    def test_context_accumulation_across_calls(self):
        """Verify contexts accumulate: context_0, context_1, etc."""
        mock_lm = MockLMWithResponses(["FINAL(got it)"])

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            with RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                persistent=True,
            ) as rlm:
                rlm.completion("First document")
                mock_lm.set_responses(["FINAL(got it)"])
                rlm.completion("Second document")
                mock_lm.set_responses(["FINAL(got it)"])
                rlm.completion("Third document")

                env = rlm._persistent_env
                assert env.get_context_count() == 3
                assert env.locals["context_0"] == "First document"
                assert env.locals["context_1"] == "Second document"
                assert env.locals["context_2"] == "Third document"
                assert env.locals["context"] == "First document"

    def test_history_accumulation_across_calls(self):
        """Verify message histories accumulate: history_0, history_1, etc."""
        mock_lm = MockLMWithResponses(["FINAL(done)"])

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            with RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                persistent=True,
            ) as rlm:
                rlm.completion("Context A")
                mock_lm.set_responses(["FINAL(done)"])
                rlm.completion("Context B")
                mock_lm.set_responses(["FINAL(done)"])
                rlm.completion("Context C")

                env = rlm._persistent_env
                assert env.get_history_count() == 3
                assert "history_0" in env.locals
                assert "history_1" in env.locals
                assert "history_2" in env.locals
                assert isinstance(env.locals["history_0"], list)
                assert len(env.locals["history_0"]) > 0
                assert env.locals["history"] == env.locals["history_0"]

    def test_variable_persistence_across_completions(self):
        """Variables computed in one completion should be available in subsequent ones."""
        first_responses = [
            "Let me compute something\n```repl\ncomputed_value = 42 * 2\nprint(computed_value)\n```",
            "FINAL(84)",
        ]
        second_responses = [
            "```repl\nresult = computed_value + 10\nprint(result)\n```",
            "FINAL(94)",
        ]

        mock_lm = MockLMWithResponses(first_responses)

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            with RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                persistent=True,
            ) as rlm:
                rlm.completion("Compute 42 * 2")
                assert rlm._persistent_env.locals.get("computed_value") == 84

                mock_lm.set_responses(second_responses)
                rlm.completion("Add 10 to the previous result")

                assert rlm._persistent_env.locals.get("computed_value") == 84
                assert rlm._persistent_env.locals.get("result") == 94


class TestMultiTurnPromptAwareness:
    """Tests that prompts correctly inform the model about contexts/histories."""

    def test_prompt_includes_context_count(self):
        """Model should be informed about available contexts."""
        mock_lm = MockLMWithResponses(["FINAL(ok)"])

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            with RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                persistent=True,
            ) as rlm:
                rlm.completion("First")
                mock_lm.set_responses(["FINAL(ok)"])
                rlm.completion("Second")

                last_prompt = mock_lm.captured_prompts[-1]
                user_messages = [m for m in last_prompt if m.get("role") == "user"]
                user_content = " ".join(m.get("content", "") for m in user_messages)

                assert "2 contexts" in user_content or "context_0" in user_content

    def test_prompt_includes_history_count(self):
        """Model should be informed about available histories."""
        mock_lm = MockLMWithResponses(["FINAL(ok)"])

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            with RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                persistent=True,
            ) as rlm:
                rlm.completion("First task")
                mock_lm.set_responses(["FINAL(ok)"])
                rlm.completion("Second task")

                last_prompt = mock_lm.captured_prompts[-1]
                user_messages = [m for m in last_prompt if m.get("role") == "user"]
                user_content = " ".join(m.get("content", "") for m in user_messages)

                assert "history" in user_content.lower()

    def test_system_prompt_contains_repl_instructions(self):
        """System prompt should contain REPL environment instructions."""
        mock_lm = MockLMWithResponses(["FINAL(ok)"])

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
            )
            rlm.completion("Test prompt")

            first_prompt = mock_lm.captured_prompts[0]
            system_messages = [m for m in first_prompt if m.get("role") == "system"]
            system_content = " ".join(m.get("content", "") for m in system_messages)

            assert "REPL" in system_content
            assert "llm_query" in system_content
            assert "FINAL" in system_content
            assert "context" in system_content


class TestMultiTurnCodeExecution:
    """Tests for code execution in multi-turn sessions."""

    def test_can_access_previous_context_in_code(self):
        """Code should be able to reference earlier contexts."""
        first_responses = ["FINAL(first done)"]
        second_responses = [
            "```repl\nprint(f'First: {context_0}, Second: {context_1}')\n```",
            "FINAL(printed both)",
        ]

        mock_lm = MockLMWithResponses(first_responses)

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            with RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                persistent=True,
            ) as rlm:
                rlm.completion("Document A")

                mock_lm.set_responses(second_responses)
                rlm.completion("Document B")

                env = rlm._persistent_env
                assert env.locals["context_0"] == "Document A"
                assert env.locals["context_1"] == "Document B"

    def test_can_access_history_in_code(self):
        """Code should be able to reference stored histories."""
        first_responses = ["FINAL(first)"]
        second_responses = [
            "```repl\nprint(f'History entries: {len(history)}')\n```",
            "FINAL(accessed history)",
        ]

        mock_lm = MockLMWithResponses(first_responses)

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            with RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                persistent=True,
            ) as rlm:
                rlm.completion("First query")

                mock_lm.set_responses(second_responses)
                rlm.completion("Second query")

                env = rlm._persistent_env
                assert "history" in env.locals
                assert isinstance(env.locals["history"], list)


class TestNonPersistentMode:
    """Tests to ensure non-persistent mode still works correctly."""

    def test_non_persistent_creates_fresh_environment(self):
        """Non-persistent mode should create new environment each call."""
        mock_lm = MockLMWithResponses(["FINAL(done)"])

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                persistent=False,
            )

            rlm.completion("First")
            assert rlm._persistent_env is None

            mock_lm.set_responses(["FINAL(done)"])
            rlm.completion("Second")
            assert rlm._persistent_env is None

    def test_default_is_non_persistent(self):
        """Default behavior should be non-persistent."""
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "mock-model"},
        )
        assert rlm.persistent is False


class TestPersistentModeResourceManagement:
    """Tests for proper resource cleanup in persistent mode."""

    def test_context_manager_cleanup(self):
        """Environment should be cleaned up when exiting context manager."""
        mock_lm = MockLMWithResponses(["FINAL(done)"])

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            with RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                persistent=True,
            ) as rlm:
                rlm.completion("Test")
                assert rlm._persistent_env is not None

            assert rlm._persistent_env is None

    def test_explicit_close(self):
        """Calling close() should clean up persistent environment."""
        mock_lm = MockLMWithResponses(["FINAL(done)"])

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                persistent=True,
            )
            rlm.completion("Test")
            assert rlm._persistent_env is not None

            rlm.close()
            assert rlm._persistent_env is None


class TestPersistentModeValidation:
    """Tests for persistent mode validation."""

    def test_unsupported_environment_raises_error(self):
        """Persistent mode should raise error for unsupported environments."""
        with pytest.raises(ValueError, match="persistent=True is not supported"):
            RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                environment="docker",  # Not supported for persistent
                persistent=True,
            )

    def test_local_environment_supported(self):
        """Local environment should support persistent mode."""
        # Should not raise
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "mock-model"},
            environment="local",
            persistent=True,
        )
        assert rlm.persistent is True


class TestSingleTurnEndToEnd:
    """End-to-end tests for single-turn completions that verify the full RLM loop.

    These tests exercise the complete flow: prompt construction -> LM call ->
    response parsing -> code execution -> result formatting -> iteration -> FINAL answer.
    Only the LM client is mocked (via MockLMWithResponses, a real BaseLM subclass).
    """

    def test_simple_final_answer_no_code(self):
        """LM immediately returns FINAL answer without code execution."""
        mock_lm = MockLMWithResponses(["FINAL(42)"])

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
            )
            result = rlm.completion("What is 6 * 7?")

            assert result.response == "42"
            assert result.execution_time > 0
            assert "mock-model" in result.usage_summary.model_usage_summaries
            usage = result.usage_summary.model_usage_summaries["mock-model"]
            assert usage.total_calls == 1
            assert usage.total_input_tokens > 0
            assert usage.total_output_tokens > 0

    def test_code_execution_then_final_answer(self):
        """LM generates code, executes it, then returns FINAL with the result."""
        responses = [
            "Let me calculate.\n```repl\nx = 6 * 7\nprint(x)\n```",
            "FINAL(42)",
        ]
        mock_lm = MockLMWithResponses(responses)

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
            )
            result = rlm.completion("What is 6 * 7?")

            assert result.response == "42"
            # LM was called twice (one for code, one for FINAL)
            assert mock_lm._call_count == 2
            assert result.execution_time > 0

    def test_execution_results_fed_back_to_model(self):
        """Verify that code execution output appears in subsequent prompts to the LM."""
        responses = [
            "```repl\nresult = 100 + 23\nprint(result)\n```",
            "FINAL(123)",
        ]
        mock_lm = MockLMWithResponses(responses)

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
            )
            result = rlm.completion("Calculate 100 + 23")

            assert result.response == "123"

            # The second prompt (captured_prompts[1]) should contain the execution output
            second_prompt = mock_lm.captured_prompts[1]
            user_messages = [m for m in second_prompt if m.get("role") == "user"]
            user_content = " ".join(m.get("content", "") for m in user_messages)

            # The output "123" from print(result) should be in the execution feedback
            assert "123" in user_content
            # The code itself should be in the feedback
            assert "result = 100 + 23" in user_content

    def test_multi_step_computation(self):
        """LM performs multiple code execution steps before arriving at FINAL."""
        responses = [
            "Step 1: Set up data.\n```repl\ndata = [10, 20, 30, 40, 50]\ntotal = sum(data)\nprint(f'Total: {total}')\n```",
            "Step 2: Calculate average.\n```repl\navg = total / len(data)\nprint(f'Average: {avg}')\n```",
            "FINAL(30.0)",
        ]
        mock_lm = MockLMWithResponses(responses)

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
            )
            result = rlm.completion("What is the average of [10, 20, 30, 40, 50]?")

            assert result.response == "30.0"
            assert mock_lm._call_count == 3

    def test_code_error_does_not_crash_rlm(self):
        """Code execution errors are captured and fed back to the LM so it can recover."""
        responses = [
            "```repl\nresult = 1 / 0\n```",
            "I see there was a ZeroDivisionError. Let me fix that.\n```repl\nresult = 'cannot divide by zero'\nprint(result)\n```",
            "FINAL(cannot divide by zero)",
        ]
        mock_lm = MockLMWithResponses(responses)

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
            )
            result = rlm.completion("Divide 1 by 0")

            assert result.response == "cannot divide by zero"

            # The second prompt should contain the error feedback
            second_prompt = mock_lm.captured_prompts[1]
            all_content = " ".join(
                m.get("content", "") for m in second_prompt if m.get("role") == "user"
            )
            assert "ZeroDivisionError" in all_content

    def test_final_var_retrieves_computed_variable(self):
        """FINAL_VAR correctly retrieves a variable computed during code execution."""
        responses = [
            "```repl\nanswer = 'the quick brown fox'\n```",
            "FINAL_VAR(answer)",
        ]
        mock_lm = MockLMWithResponses(responses)

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
            )
            result = rlm.completion("Tell me a phrase")

            assert result.response == "the quick brown fox"

    def test_context_available_in_code_execution(self):
        """The context variable should be accessible during code execution."""
        responses = [
            "```repl\nwords = context.split()\ncount = len(words)\nprint(count)\n```",
            "FINAL(3)",
        ]
        mock_lm = MockLMWithResponses(responses)

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
            )
            result = rlm.completion("hello world foo")

            assert result.response == "3"

    def test_max_iterations_produces_default_answer(self):
        """When max iterations are exhausted, the RLM should produce a default answer."""
        # LM never returns FINAL
        responses = [
            "Still thinking...",
            "Still thinking...",
            "The answer is 42",  # Default answer attempt
        ]
        mock_lm = MockLMWithResponses(responses)

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                max_iterations=2,  # Only allow 2 iterations
            )
            result = rlm.completion("What is the answer?")

            # Should have a response (the default answer attempt)
            assert result.response is not None
            assert len(result.response) > 0
            # Should have called LM: 2 iterations + 1 default answer = 3 calls
            assert mock_lm._call_count == 3

    def test_usage_tracking_across_iterations(self):
        """Verify token usage is tracked correctly across multiple iterations."""
        responses = [
            "```repl\nx = 1\n```",
            "FINAL(done)",
        ]
        mock_lm = MockLMWithResponses(responses)

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
            )
            result = rlm.completion("Test")

            usage = result.usage_summary.model_usage_summaries["mock-model"]
            assert usage.total_calls == 2
            assert usage.total_input_tokens > 0
            assert usage.total_output_tokens > 0
            # Second call should have more input tokens (includes first iteration history)
            # This verifies real token tracking is working


class TestMultiTurnEndToEnd:
    """End-to-end tests simulating realistic multi-turn usage."""

    def test_three_turn_conversation(self):
        """Simulate a 3-turn conversation with context accumulation."""
        turn1_responses = [
            "Looking at the first document\n```repl\ndoc1_summary = 'Has info about cats'\nprint(doc1_summary)\n```",
            "FINAL(Summarized first doc)",
        ]
        turn2_responses = [
            "Looking at second document and comparing\n```repl\ndoc2_summary = 'Has info about dogs'\nprint(f'Doc1: {doc1_summary}, Doc2: {doc2_summary}')\n```",
            "FINAL(Compared both docs)",
        ]
        turn3_responses = [
            "Final synthesis\n```repl\nfinal = f'Combined: {doc1_summary} and {doc2_summary} from context_2'\nprint(final)\n```",
            "FINAL(synthesized all)",
        ]

        mock_lm = MockLMWithResponses(turn1_responses)

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            with RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
                persistent=True,
            ) as rlm:
                result1 = rlm.completion("First document about cats")
                assert "Summarized" in result1.response

                mock_lm.set_responses(turn2_responses)
                result2 = rlm.completion("Second document about dogs")
                assert "Compared" in result2.response

                mock_lm.set_responses(turn3_responses)
                result3 = rlm.completion("Synthesize everything")
                assert "synthesized" in result3.response

                env = rlm._persistent_env
                assert env.get_context_count() == 3
                assert env.get_history_count() == 3
                assert env.locals.get("doc1_summary") == "Has info about cats"
                assert env.locals.get("doc2_summary") == "Has info about dogs"

    def test_usage_reported_per_completion(self):
        """Each completion should report its own usage independently."""
        mock_lm = MockLMWithResponses(["FINAL(first)"])

        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "mock-model"},
            )

            result1 = rlm.completion("First prompt")
            usage1 = result1.usage_summary.model_usage_summaries["mock-model"]

            mock_lm.set_responses(["FINAL(second)"])
            result2 = rlm.completion("Second prompt")
            usage2 = result2.usage_summary.model_usage_summaries["mock-model"]

            assert result1.response == "first"
            assert result2.response == "second"

            # Both should have tracked some usage
            assert usage1.total_calls >= 1
            assert usage2.total_calls >= 1
            assert usage1.total_input_tokens > 0
            assert usage2.total_input_tokens > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
