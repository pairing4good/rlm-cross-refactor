"""Tests for backend_kwargs validation to prevent common token limit mistakes."""

import pytest

from rlm import RLM


class TestBackendKwargsValidation:
    """Tests for validating backend_kwargs to catch token limit configuration errors."""

    def test_max_root_tokens_in_backend_kwargs_raises_error(self):
        """Should raise error if max_root_tokens is in backend_kwargs."""
        with pytest.raises(ValueError, match="max_root_tokens.*backend_kwargs"):
            RLM(
                backend="openai",
                backend_kwargs={
                    "model_name": "gpt-4o",
                    "max_root_tokens": 1000,  # Wrong place
                },
            )

    def test_max_sub_tokens_in_other_backend_kwargs_raises_error(self):
        """Should raise error if max_sub_tokens is in other_backend_kwargs."""
        with pytest.raises(ValueError, match="max_sub_tokens.*other_backend_kwargs"):
            RLM(
                backend="openai",
                backend_kwargs={
                    "model_name": "gpt-4o",
                },
                other_backends=["openai"],
                other_backend_kwargs=[
                    {
                        "model_name": "gpt-3.5-turbo",
                        "max_sub_tokens": 1000,  # Wrong place
                    }
                ],
            )

    def test_low_max_tokens_in_backend_kwargs_raises_error(self):
        """Should raise error if max_tokens < 100 in backend_kwargs."""
        with pytest.raises(ValueError, match="max_tokens.*unreasonably low"):
            RLM(
                backend="openai",
                backend_kwargs={
                    "model_name": "gpt-4o",
                    "max_tokens": 1,  # Too low - would cause 1-token responses
                },
            )

    def test_low_max_tokens_in_other_backend_kwargs_raises_error(self):
        """Should raise error if max_tokens < 100 in other_backend_kwargs."""
        with pytest.raises(ValueError, match="max_tokens.*unreasonably low"):
            RLM(
                backend="openai",
                backend_kwargs={"model_name": "gpt-4o"},
                other_backends=["openai"],
                other_backend_kwargs=[
                    {
                        "model_name": "gpt-3.5-turbo",
                        "max_tokens": 50,  # Too low
                    }
                ],
            )

    def test_reasonable_max_tokens_in_backend_kwargs_allowed(self):
        """Should allow reasonable max_tokens values (>= 100)."""
        # This should not raise
        rlm = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": "gpt-4o",
                "max_tokens": 1000,  # Reasonable per-response limit
            },
        )
        assert rlm is not None
        assert rlm.backend_kwargs["max_tokens"] == 1000

    def test_no_max_tokens_in_backend_kwargs_allowed(self):
        """Should allow backend_kwargs without any max_tokens parameter."""
        # This should not raise
        rlm = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": "gpt-4o",
            },
            max_root_tokens=100000,  # Correct place for session limit
        )
        assert rlm is not None
        assert "max_tokens" not in rlm.backend_kwargs

    def test_constructor_token_limits_work_correctly(self):
        """Verify constructor token limit parameters work as expected."""
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "gpt-4o"},
            max_root_tokens=50000,
            max_sub_tokens=25000,
        )

        assert rlm.max_root_tokens == 50000
        assert rlm.max_sub_tokens == 25000
        assert "max_root_tokens" not in rlm.backend_kwargs
        assert "max_sub_tokens" not in rlm.backend_kwargs

    def test_none_token_limits_allowed(self):
        """Should allow None token limits (unlimited)."""
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "gpt-4o"},
            max_root_tokens=None,
            max_sub_tokens=None,
        )

        assert rlm.max_root_tokens is None
        assert rlm.max_sub_tokens is None

    def test_empty_backend_kwargs_allowed(self):
        """Should allow None or empty backend_kwargs."""
        rlm1 = RLM(backend="openai", backend_kwargs=None)
        assert rlm1 is not None

        rlm2 = RLM(backend="openai", backend_kwargs={})
        assert rlm2 is not None

    def test_multiple_other_backends_validates_all(self):
        """Should validate all entries in other_backend_kwargs."""
        # This test verifies validation runs on all entries
        # (Even though only 1 backend is currently supported, validation should work for future)
        with pytest.raises(ValueError, match="max_sub_tokens.*other_backend_kwargs\\[0\\]"):
            RLM(
                backend="openai",
                backend_kwargs={"model_name": "gpt-4o"},
                other_backends=["openai"],
                other_backend_kwargs=[
                    {
                        "model_name": "gpt-3.5-turbo",
                        "max_sub_tokens": 1000,  # Wrong place
                    }
                ],
            )

    def test_error_message_includes_example(self):
        """Error messages should include helpful examples."""
        with pytest.raises(ValueError) as exc_info:
            RLM(
                backend="openai",
                backend_kwargs={
                    "model_name": "gpt-4o",
                    "max_root_tokens": 1000,
                },
            )

        error_msg = str(exc_info.value)
        assert "Example:" in error_msg
        assert "RLM(" in error_msg
        assert "max_root_tokens=100000" in error_msg

    def test_low_max_tokens_error_message_helpful(self):
        """Error message for low max_tokens should be helpful."""
        with pytest.raises(ValueError) as exc_info:
            RLM(
                backend="openai",
                backend_kwargs={
                    "model_name": "gpt-4o",
                    "max_tokens": 1,
                },
            )

        error_msg = str(exc_info.value)
        assert "unreasonably low" in error_msg
        assert "incomplete responses" in error_msg
        assert "at least 100 tokens" in error_msg
