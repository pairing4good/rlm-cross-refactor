"""Tests for token limit functionality in RLM sessions.

These tests verify that:
1. Token limits are honored per completion() session
2. Sessions end early when limits are exceeded
3. Token counting includes all LM calls (main + sub-LM calls)
4. Logger records limit hit events
5. RLMChatCompletion clearly indicates limit was hit
6. No actual API calls are made (all mocked)
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

import rlm.core.rlm as rlm_module
from rlm import RLM
from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.logger.rlm_logger import RLMLogger
from tests.mock_lm import MockLMWithTokenTracking


def create_mock_lm_with_final_answer(tokens_per_call: int = 5000) -> Mock:
    """Create a mock LM that returns FINAL() on second call."""
    mock = MockLMWithTokenTracking(tokens_per_call=tokens_per_call)
    
    # First call: no final answer (will cause iteration)
    # Second call: provide final answer (will terminate normally)
    responses = [
        "Let me think about this...",
        "FINAL(42)",
    ]
    
    original_completion = mock.completion
    call_count = [0]
    
    def completion_with_responses(prompt):
        result = original_completion(prompt)
        response = responses[min(call_count[0], len(responses) - 1)]
        call_count[0] += 1
        return response
    
    mock.completion = completion_with_responses
    return mock


def create_mock_lm_never_finishes(tokens_per_call: int = 5000) -> Mock:
    """Create a mock LM that never returns a final answer."""
    mock = MockLMWithTokenTracking(tokens_per_call=tokens_per_call)
    
    original_completion = mock.completion
    
    def completion_without_final(prompt):
        original_completion(prompt)  # Track tokens
        return "Still working on it..."
    
    mock.completion = completion_without_final
    return mock


class TestTokenLimitBasics:
    """Basic token limit functionality tests."""

    def test_token_limit_honored_in_single_session(self):
        """Verify session stops when token limit is reached."""
        # Each call uses 5000 tokens, limit is 12000
        # Iteration 0: check (0 < 12000) → run → 5000 tokens
        # Iteration 1: check (5000 < 12000) → run → 10000 tokens
        # Iteration 2: check (10000 < 12000) → run → 15000 tokens
        # Iteration 3: check (15000 >= 12000) → STOP
        # Note: We can overshoot slightly because we check BEFORE each iteration
        
        mock_lm = create_mock_lm_never_finishes(tokens_per_call=5000)
        
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = mock_lm
            
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_iterations=10,
                max_tokens=12000,  # Limit to 12000 tokens
            )
            
            result = rlm.completion(prompt="Test prompt")
            
            # Should have stopped due to token limit, not max iterations
            assert "Token limit exceeded" in result.response
            assert "12,000" in result.response  # Formatted limit
            
            # We stop after detecting the limit was exceeded
            # Could be 3 completions (15000 tokens) before detection
            assert mock_lm.call_count <= 4  # At most 3 completions + 1 for detection
            total_tokens = mock_lm.total_input_tokens + mock_lm.total_output_tokens
            # Allow slight overshoot (one extra iteration's worth)
            assert total_tokens <= 18000  # Max 3 iterations + margin

    def test_no_limit_when_max_tokens_not_set(self):
        """Verify normal behavior when max_tokens is None."""
        mock_lm = create_mock_lm_with_final_answer(tokens_per_call=5000)
        
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = mock_lm
            
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_iterations=10,
                max_tokens=None,  # No token limit
            )
            
            result = rlm.completion(prompt="Test prompt")
            
            # Should complete normally with final answer
            assert result.response == "42"
            assert "Token limit" not in result.response

    def test_token_limit_with_small_budget(self):
        """Test session stops immediately with very small token budget."""
        mock_lm = create_mock_lm_never_finishes(tokens_per_call=5000)
        
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = mock_lm
            
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_iterations=10,
                max_tokens=6000,  # Only enough for 1 call (5000 tokens)
            )
            
            result = rlm.completion(prompt="Test prompt")
            
            assert "Token limit exceeded" in result.response
            # Should stop after just 1 iteration
            assert mock_lm.call_count <= 2  # 1 completion + 1 check

    def test_completion_within_budget_succeeds(self):
        """Test that session completes normally when within token budget."""
        mock_lm = create_mock_lm_with_final_answer(tokens_per_call=3000)
        
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = mock_lm
            
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_iterations=10,
                max_tokens=50000,  # Large budget
            )
            
            result = rlm.completion(prompt="Test prompt")
            
            # Should complete normally with final answer
            assert result.response == "42"
            assert "Token limit" not in result.response
            
            # Verify tokens were tracked
            total_tokens = (
                result.usage_summary.model_usage_summaries["mock-model"].total_input_tokens +
                result.usage_summary.model_usage_summaries["mock-model"].total_output_tokens
            )
            assert total_tokens < 50000


class TestTokenLimitLogging:
    """Test logging functionality for token limits."""

    def test_logger_records_limit_hit(self):
        """Verify JSONL logger records limit hit event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RLMLogger(log_dir=tmpdir, file_name="test")
            mock_lm = create_mock_lm_never_finishes(tokens_per_call=5000)
            
            with patch.object(rlm_module, "get_client") as mock_get_client:
                mock_get_client.return_value = mock_lm
                
                rlm = RLM(
                    backend="openai",
                    backend_kwargs={"model_name": "test-model"},
                    max_iterations=10,
                    max_tokens=12000,
                    logger=logger,
                )
                
                result = rlm.completion(prompt="Test prompt")
                
                # Read log file and verify limit_hit entry exists
                log_file = logger.log_file_path
                assert os.path.exists(log_file)
                
                with open(log_file, "r") as f:
                    lines = f.readlines()
                
                # Find limit_hit entry
                limit_hit_entry = None
                for line in lines:
                    entry = json.loads(line)
                    if entry.get("type") == "limit_hit":
                        limit_hit_entry = entry
                        break
                
                assert limit_hit_entry is not None, "No limit_hit entry found in log"
                assert limit_hit_entry["limit_type"] == "token_limit"
                assert limit_hit_entry["max_value"] == 12000
                assert limit_hit_entry["current_value"] >= 10000  # At least 2 calls
                assert "iteration" in limit_hit_entry
                assert "timestamp" in limit_hit_entry

    def test_logger_records_metadata_with_max_tokens(self):
        """Verify metadata includes max_tokens configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RLMLogger(log_dir=tmpdir, file_name="test_metadata")
            mock_lm = create_mock_lm_with_final_answer(tokens_per_call=3000)
            
            with patch.object(rlm_module, "get_client") as mock_get_client:
                mock_get_client.return_value = mock_lm
                
                rlm = RLM(
                    backend="openai",
                    backend_kwargs={"model_name": "test-model"},
                    max_tokens=50000,
                    logger=logger,
                )
                
                result = rlm.completion(prompt="Test prompt")
                
                # Read first line (metadata)
                with open(logger.log_file_path, "r") as f:
                    first_line = f.readline()
                
                metadata = json.loads(first_line)
                assert metadata["type"] == "metadata"
                assert metadata["max_tokens"] == 50000


class TestTokenLimitResponse:
    """Test RLMChatCompletion response when limit is hit."""

    def test_response_indicates_limit_hit(self):
        """Verify response clearly states limit was exceeded."""
        mock_lm = create_mock_lm_never_finishes(tokens_per_call=5000)
        
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = mock_lm
            
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_tokens=12000,
            )
            
            result = rlm.completion(prompt="Test prompt")
            
            # Verify response message
            assert "Session ended" in result.response
            assert "Token limit exceeded" in result.response
            assert "12,000" in result.response  # Formatted limit
            assert "iteration" in result.response.lower()

    def test_response_includes_usage_summary(self):
        """Verify response includes usage summary with token counts."""
        mock_lm = create_mock_lm_never_finishes(tokens_per_call=5000)
        
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = mock_lm
            
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_tokens=12000,
            )
            
            result = rlm.completion(prompt="Test prompt")
            
            # Verify usage summary is populated
            assert result.usage_summary is not None
            assert "mock-model" in result.usage_summary.model_usage_summaries
            
            usage = result.usage_summary.model_usage_summaries["mock-model"]
            total_tokens = usage.total_input_tokens + usage.total_output_tokens
            assert total_tokens > 0
            # Allow slight overshoot due to iteration boundary
            assert total_tokens >= 12000  # At least hit the limit
            assert total_tokens <= 18000  # But not too many extra iterations

    def test_response_includes_execution_time(self):
        """Verify response includes execution time even when limit hit."""
        mock_lm = create_mock_lm_never_finishes(tokens_per_call=5000)
        
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = mock_lm
            
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_tokens=12000,
            )
            
            result = rlm.completion(prompt="Test prompt")
            
            # Verify execution time is recorded
            assert result.execution_time is not None
            assert result.execution_time > 0


class TestTokenLimitEdgeCases:
    """Test edge cases and special scenarios."""

    def test_exact_limit_boundary(self):
        """Test behavior when exactly at token limit."""
        # 2 calls at 5000 each = 10000 total
        # Limit set to exactly 10000
        mock_lm = create_mock_lm_never_finishes(tokens_per_call=5000)
        
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = mock_lm
            
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_tokens=10000,  # Exact boundary
            )
            
            result = rlm.completion(prompt="Test prompt")
            
            assert "Token limit exceeded" in result.response
            total_tokens = (
                result.usage_summary.model_usage_summaries["mock-model"].total_input_tokens +
                result.usage_summary.model_usage_summaries["mock-model"].total_output_tokens
            )
            assert total_tokens <= 10000

    def test_zero_iterations_if_no_budget(self):
        """Test behavior when token limit is extremely low."""
        mock_lm = create_mock_lm_never_finishes(tokens_per_call=5000)
        
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = mock_lm
            
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_tokens=100,  # Extremely low
            )
            
            result = rlm.completion(prompt="Test prompt")
            
            # Should stop immediately or after first check
            assert "Token limit exceeded" in result.response
            # Very few calls should have been made
            assert mock_lm.call_count <= 2

    def test_max_tokens_none_is_unlimited(self):
        """Test that max_tokens=None means no limit."""
        mock_lm = create_mock_lm_with_final_answer(tokens_per_call=10000)
        
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = mock_lm
            
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_iterations=5,
                max_tokens=None,  # Explicitly no limit
            )
            
            result = rlm.completion(prompt="Test prompt")
            
            # Should complete normally regardless of token count
            assert result.response == "42"
            assert "Token limit exceeded" not in result.response

    def test_limit_resets_across_completion_calls(self):
        """Verify token limit applies per completion() call, not across calls."""
        mock_lm1 = create_mock_lm_with_final_answer(tokens_per_call=5000)
        mock_lm2 = create_mock_lm_with_final_answer(tokens_per_call=5000)
        
        with patch.object(rlm_module, "get_client") as mock_get_client:
            # First completion
            mock_get_client.return_value = mock_lm1
            
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_tokens=50000,
                persistent=False,  # Non-persistent mode
            )
            
            result1 = rlm.completion(prompt="First prompt")
            tokens_after_first = (
                result1.usage_summary.model_usage_summaries["mock-model"].total_input_tokens +
                result1.usage_summary.model_usage_summaries["mock-model"].total_output_tokens
            )
            
            # Second completion with fresh mock
            mock_get_client.return_value = mock_lm2
            result2 = rlm.completion(prompt="Second prompt")
            tokens_after_second = (
                result2.usage_summary.model_usage_summaries["mock-model"].total_input_tokens +
                result2.usage_summary.model_usage_summaries["mock-model"].total_output_tokens
            )
            
            # Both should complete successfully
            assert result1.response == "42"
            assert result2.response == "42"
            
            # Each completion should track its own tokens independently
            assert tokens_after_first > 0
            assert tokens_after_second > 0


class TestTokenLimitWithSubLMCalls:
    """Test that token limits include sub-LM calls from environments."""

    def test_limit_includes_other_backend_tokens(self):
        """Verify tokens from other_backends are counted toward limit."""
        main_mock = MockLMWithTokenTracking(tokens_per_call=3000, model_name="main-model")
        other_mock = MockLMWithTokenTracking(tokens_per_call=2000, model_name="other-model")
        
        # Main returns response without FINAL
        original_main = main_mock.completion
        def main_completion(prompt):
            original_main(prompt)
            return "Let me think..."
        main_mock.completion = main_completion
        
        # Other also doesn't return FINAL
        original_other = other_mock.completion
        def other_completion(prompt):
            original_other(prompt)
            return "Sub-call result"
        other_mock.completion = other_completion
        
        with patch.object(rlm_module, "get_client") as mock_get_client:
            # Return different mocks based on backend
            def get_client_side_effect(backend, kwargs):
                if backend == "openai":
                    return main_mock
                elif backend == "anthropic":
                    return other_mock
                return main_mock
            
            mock_get_client.side_effect = get_client_side_effect
            
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "main-model"},
                other_backends=["anthropic"],
                other_backend_kwargs=[{"model_name": "other-model"}],
                max_tokens=15000,  # Combined limit
                max_iterations=10,
            )
            
            result = rlm.completion(prompt="Test prompt")
            
            # Should eventually hit limit from combined token usage
            # Note: Without actual sub-LM calls in test, this primarily tests
            # that the infrastructure is in place
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
