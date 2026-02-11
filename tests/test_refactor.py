"""Tests for the refactoring module: DockerREPL working_dir support and CLI."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from rlm.environments.docker_repl import _build_exec_script
from rlm.refactor.cli import (
    INSTRUCTIONS_FILENAME,
    build_backend_kwargs,
    parse_arg_value,
    parse_args,
)
from rlm.refactor.prompts import REFACTOR_SYSTEM_PROMPT


class TestBuildExecScriptWorkingDir:
    """Tests for _build_exec_script with working_dir parameter."""

    def test_no_working_dir_unchanged(self):
        """When working_dir is None, script should not contain /repos chdir or extra imports."""
        script = _build_exec_script("x = 1", proxy_port=8080, depth=1, working_dir=None)
        assert "os.chdir('/repos')" not in script
        assert "import subprocess" not in script
        assert "import glob" not in script

    def test_working_dir_adds_chdir(self):
        """When working_dir is set, script should chdir to /repos."""
        script = _build_exec_script("x = 1", proxy_port=8080, depth=1, working_dir="/tmp/repos")
        assert "os.chdir('/repos')" in script

    def test_working_dir_adds_extra_imports(self):
        """When working_dir is set, script should import subprocess, glob, shutil, pathlib."""
        script = _build_exec_script("x = 1", proxy_port=8080, depth=1, working_dir="/tmp/repos")
        assert "subprocess" in script
        assert "glob" in script
        assert "shutil" in script
        assert "pathlib" in script

    def test_code_still_embedded(self):
        """User code is still base64-embedded regardless of working_dir."""
        script = _build_exec_script(
            "print('hello')", proxy_port=8080, depth=1, working_dir="/tmp/repos"
        )
        assert "base64.b64decode" in script

    def test_proxy_port_in_script(self):
        """Proxy port appears in the script URL."""
        script = _build_exec_script("x = 1", proxy_port=9999, depth=1, working_dir="/tmp/repos")
        assert "9999" in script


class TestDockerREPLWorkingDir:
    """Tests for DockerREPL constructor and setup with working_dir."""

    @patch("rlm.environments.docker_repl.subprocess.run")
    @patch("rlm.environments.docker_repl.HTTPServer")
    def test_working_dir_adds_volume_mount(self, mock_httpserver, mock_run):
        """When working_dir is set, docker run should include -v for /repos."""
        from rlm.environments.docker_repl import DockerREPL

        # Mock HTTPServer
        mock_server = MagicMock()
        mock_server.server_address = ("127.0.0.1", 12345)
        mock_httpserver.return_value = mock_server

        # Mock all subprocess.run calls to succeed
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "container123\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = os.path.join(tmpdir, "repos")
            os.makedirs(working_dir)

            repl = DockerREPL(working_dir=working_dir)

            # Find the docker run call
            docker_run_calls = [
                call
                for call in mock_run.call_args_list
                if call[0] and call[0][0][:2] == ["docker", "run"]
            ]
            assert len(docker_run_calls) == 1

            docker_cmd = docker_run_calls[0][0][0]
            # Check that -v working_dir:/repos is in the command
            assert "-v" in docker_cmd
            repos_mount_idx = None
            for i, arg in enumerate(docker_cmd):
                if arg == "-v" and i + 1 < len(docker_cmd) and ":/repos" in docker_cmd[i + 1]:
                    repos_mount_idx = i
                    break
            assert repos_mount_idx is not None, f"No /repos mount found in: {docker_cmd}"

            repl.cleanup()

    @patch("rlm.environments.docker_repl.subprocess.run")
    @patch("rlm.environments.docker_repl.HTTPServer")
    def test_working_dir_installs_git(self, mock_httpserver, mock_run):
        """When working_dir is set, setup should install git and configure safe.directory."""
        from rlm.environments.docker_repl import DockerREPL

        mock_server = MagicMock()
        mock_server.server_address = ("127.0.0.1", 12345)
        mock_httpserver.return_value = mock_server

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "container123\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = os.path.join(tmpdir, "repos")
            os.makedirs(working_dir)

            repl = DockerREPL(working_dir=working_dir)

            # Check that git install was called
            all_calls_str = str(mock_run.call_args_list)
            assert "apt-get" in all_calls_str and "git" in all_calls_str

            # Check that git config safe.directory was called
            assert "safe.directory" in all_calls_str

            repl.cleanup()

    @patch("rlm.environments.docker_repl.subprocess.run")
    @patch("rlm.environments.docker_repl.HTTPServer")
    def test_no_working_dir_no_git(self, mock_httpserver, mock_run):
        """When working_dir is None, setup should NOT install git."""
        from rlm.environments.docker_repl import DockerREPL

        mock_server = MagicMock()
        mock_server.server_address = ("127.0.0.1", 12345)
        mock_httpserver.return_value = mock_server

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "container123\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        repl = DockerREPL()

        # Should NOT have git install calls
        all_calls_str = str(mock_run.call_args_list)
        assert "apt-get" not in all_calls_str
        assert "safe.directory" not in all_calls_str

        repl.cleanup()

    @patch("rlm.environments.docker_repl.subprocess.run")
    @patch("rlm.environments.docker_repl.HTTPServer")
    def test_no_working_dir_no_repos_mount(self, mock_httpserver, mock_run):
        """When working_dir is None, docker run should NOT include /repos mount."""
        from rlm.environments.docker_repl import DockerREPL

        mock_server = MagicMock()
        mock_server.server_address = ("127.0.0.1", 12345)
        mock_httpserver.return_value = mock_server

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "container123\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        repl = DockerREPL()

        docker_run_calls = [
            call
            for call in mock_run.call_args_list
            if call[0] and call[0][0][:2] == ["docker", "run"]
        ]
        docker_cmd = docker_run_calls[0][0][0]

        for i, arg in enumerate(docker_cmd):
            if arg == "-v" and i + 1 < len(docker_cmd):
                assert ":/repos" not in docker_cmd[i + 1], "Should not have /repos mount"

        repl.cleanup()


class TestRefactorCLIParsing:
    """Tests for CLI argument parsing."""

    def test_required_args(self):
        args = parse_args(["--working-dir", "/tmp/repos", "--model", "gpt-4o"])
        assert args.working_dir == "/tmp/repos"
        assert args.model == "gpt-4o"
        assert args.backend == "openai"
        assert args.max_iterations == 50
        assert args.verbose is False

    def test_all_args(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp/repos",
                "--model",
                "gpt-4o",
                "--backend",
                "anthropic",
                "--sub-backend",
                "openai",
                "--sub-model",
                "gpt-4o-mini",
                "--max-iterations",
                "100",
                "--image",
                "my-custom-image",
                "--verbose",
            ]
        )
        assert args.backend == "anthropic"
        assert args.sub_backend == "openai"
        assert args.sub_model == "gpt-4o-mini"
        assert args.max_iterations == 100
        assert args.image == "my-custom-image"
        assert args.verbose is True

    def test_missing_required_args(self):
        with pytest.raises(SystemExit):
            parse_args(["--working-dir", "/tmp/repos"])  # missing --model

        with pytest.raises(SystemExit):
            parse_args(["--model", "gpt-4o"])  # missing --working-dir


class TestRefactorCLIValidation:
    """Tests for CLI validation logic."""

    @patch("rlm.refactor.cli.RLM")
    def test_missing_instructions_file(self, mock_rlm):
        """CLI should exit with error if rlm-instructions.md is missing."""
        from rlm.refactor.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(SystemExit) as exc_info:
                main(["--working-dir", tmpdir, "--model", "gpt-4o"])
            assert exc_info.value.code == 1

    @patch("rlm.refactor.cli.RLM")
    def test_nonexistent_working_dir(self, mock_rlm):
        """CLI should exit with error if working dir doesn't exist."""
        from rlm.refactor.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--working-dir", "/nonexistent/path/xyz", "--model", "gpt-4o"])
        assert exc_info.value.code == 1

    @patch("rlm.refactor.cli.RLM")
    def test_empty_instructions_file(self, mock_rlm):
        """CLI should exit with error if rlm-instructions.md is empty."""
        from rlm.refactor.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, INSTRUCTIONS_FILENAME), "w") as f:
                f.write("")
            with pytest.raises(SystemExit) as exc_info:
                main(["--working-dir", tmpdir, "--model", "gpt-4o"])
            assert exc_info.value.code == 1

    @patch("rlm.refactor.cli.RLM")
    def test_creates_log_dir(self, mock_rlm):
        """CLI should create rlm-logs/ directory in working dir."""
        from rlm.refactor.cli import main

        # Mock the RLM to avoid actual API calls
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.response = "Done"
        mock_result.execution_time = 1.0
        mock_instance.completion.return_value = mock_result
        mock_rlm.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, INSTRUCTIONS_FILENAME), "w") as f:
                f.write("Rename foo to bar")

            main(["--working-dir", tmpdir, "--model", "gpt-4o"])

            assert os.path.isdir(os.path.join(tmpdir, "rlm-logs"))

    @patch("rlm.refactor.cli.RLM")
    def test_passes_working_dir_to_rlm(self, mock_rlm):
        """CLI should pass working_dir in environment_kwargs."""
        from rlm.refactor.cli import main

        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.response = "Done"
        mock_result.execution_time = 1.0
        mock_instance.completion.return_value = mock_result
        mock_rlm.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, INSTRUCTIONS_FILENAME), "w") as f:
                f.write("Rename foo to bar")

            main(["--working-dir", tmpdir, "--model", "gpt-4o"])

            call_kwargs = mock_rlm.call_args[1]
            assert call_kwargs["environment"] == "docker"
            assert "working_dir" in call_kwargs["environment_kwargs"]
            assert call_kwargs["environment_kwargs"]["working_dir"] == os.path.abspath(tmpdir)


class TestParseArgValue:
    """Tests for parse_arg_value helper function."""

    def test_parse_string(self):
        key, value = parse_arg_value("base_url=http://localhost:8080")
        assert key == "base_url"
        assert value == "http://localhost:8080"
        assert isinstance(value, str)

    def test_parse_int(self):
        key, value = parse_arg_value("max_tokens=4000")
        assert key == "max_tokens"
        assert value == 4000
        assert isinstance(value, int)

    def test_parse_float(self):
        key, value = parse_arg_value("temperature=0.7")
        assert key == "temperature"
        assert value == 0.7
        assert isinstance(value, float)

    def test_parse_bool_true(self):
        key, value = parse_arg_value("stream=true")
        assert key == "stream"
        assert value is True

    def test_parse_bool_false(self):
        key, value = parse_arg_value("stream=false")
        assert key == "stream"
        assert value is False

    def test_parse_with_equals_in_value(self):
        key, value = parse_arg_value("url=http://example.com?foo=bar")
        assert key == "url"
        assert value == "http://example.com?foo=bar"

    def test_missing_equals_raises_error(self):
        with pytest.raises(ValueError, match="Expected KEY=VALUE"):
            parse_arg_value("invalid_arg")


class TestBuildBackendKwargs:
    """Tests for build_backend_kwargs function."""

    def test_root_model_only(self):
        args = parse_args(["--working-dir", "/tmp", "--model", "gpt-4o"])
        kwargs = build_backend_kwargs(args, is_sub=False)
        assert kwargs == {"model_name": "gpt-4o"}

    def test_root_with_max_tokens(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--max-tokens",
                "8000",
            ]
        )
        kwargs = build_backend_kwargs(args, is_sub=False)
        assert kwargs["model_name"] == "gpt-4o"
        assert kwargs["max_tokens"] == 8000

    def test_root_with_temperature_and_top_p(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--temperature",
                "0.7",
                "--top-p",
                "0.95",
            ]
        )
        kwargs = build_backend_kwargs(args, is_sub=False)
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.95

    def test_root_with_backend_arg(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--backend-arg",
                "base_url=http://localhost:8080",
                "--backend-arg",
                "api_key=sk-test",
            ]
        )
        kwargs = build_backend_kwargs(args, is_sub=False)
        assert kwargs["base_url"] == "http://localhost:8080"
        assert kwargs["api_key"] == "sk-test"

    def test_sub_model_basic(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--sub-model",
                "gpt-4o-mini",
            ]
        )
        kwargs = build_backend_kwargs(args, is_sub=True)
        assert kwargs == {"model_name": "gpt-4o-mini"}

    def test_sub_model_with_params(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--sub-model",
                "gpt-4o-mini",
                "--sub-max-tokens",
                "4000",
                "--sub-temperature",
                "0.3",
                "--sub-top-p",
                "0.9",
            ]
        )
        kwargs = build_backend_kwargs(args, is_sub=True)
        assert kwargs["model_name"] == "gpt-4o-mini"
        assert kwargs["max_tokens"] == 4000
        assert kwargs["temperature"] == 0.3
        assert kwargs["top_p"] == 0.9

    def test_sub_backend_arg(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--sub-model",
                "gpt-4o-mini",
                "--sub-backend-arg",
                "base_url=http://localhost:9000",
            ]
        )
        kwargs = build_backend_kwargs(args, is_sub=True)
        assert kwargs["base_url"] == "http://localhost:9000"

    def test_root_and_sub_dont_interfere(self):
        """Root params shouldn't affect sub kwargs and vice versa."""
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--max-tokens",
                "8000",
                "--temperature",
                "0.7",
                "--sub-model",
                "gpt-4o-mini",
                "--sub-max-tokens",
                "4000",
                "--sub-temperature",
                "0.3",
            ]
        )
        root_kwargs = build_backend_kwargs(args, is_sub=False)
        sub_kwargs = build_backend_kwargs(args, is_sub=True)

        assert root_kwargs["max_tokens"] == 8000
        assert root_kwargs["temperature"] == 0.7

        assert sub_kwargs["max_tokens"] == 4000
        assert sub_kwargs["temperature"] == 0.3


class TestEnhancedCLIParsing:
    """Tests for enhanced CLI argument parsing."""

    def test_max_tokens_args(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--max-tokens",
                "8000",
                "--sub-model",
                "gpt-4o-mini",
                "--sub-max-tokens",
                "4000",
            ]
        )
        assert args.max_tokens == 8000
        assert args.sub_max_tokens == 4000

    def test_temperature_args(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--temperature",
                "0.7",
                "--sub-temperature",
                "0.3",
            ]
        )
        assert args.temperature == 0.7
        assert args.sub_temperature == 0.3

    def test_top_p_args(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--top-p",
                "0.95",
                "--sub-top-p",
                "0.9",
            ]
        )
        assert args.top_p == 0.95
        assert args.sub_top_p == 0.9

    def test_backend_arg_parsing(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--backend-arg",
                "base_url=http://localhost:8080",
                "--backend-arg",
                "timeout=30",
            ]
        )
        assert args.backend_arg == ["base_url=http://localhost:8080", "timeout=30"]

    def test_max_root_tokens_default(self):
        args = parse_args(["--working-dir", "/tmp", "--model", "gpt-4o"])
        assert args.max_root_tokens == 1_000_000

    def test_max_root_tokens_custom(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--max-root-tokens",
                "500000",
            ]
        )
        assert args.max_root_tokens == 500000

    def test_max_sub_tokens_custom(self):
        args = parse_args(
            [
                "--working-dir",
                "/tmp",
                "--model",
                "gpt-4o",
                "--max-sub-tokens",
                "200000",
            ]
        )
        assert args.max_sub_tokens == 200000


class TestEnhancedCLIValidation:
    """Tests for CLI validation with new parameters."""

    @patch("rlm.refactor.cli.RLM")
    def test_backend_kwargs_passed_to_rlm(self, mock_rlm):
        """CLI should pass all backend_kwargs to RLM."""
        from rlm.refactor.cli import main

        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.response = "Done"
        mock_result.execution_time = 1.0
        mock_instance.completion.return_value = mock_result
        mock_rlm.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, INSTRUCTIONS_FILENAME), "w") as f:
                f.write("Test task")

            main(
                [
                    "--working-dir",
                    tmpdir,
                    "--model",
                    "gpt-4o",
                    "--max-tokens",
                    "8000",
                    "--temperature",
                    "0.7",
                ]
            )

            call_kwargs = mock_rlm.call_args[1]
            assert call_kwargs["backend_kwargs"]["model_name"] == "gpt-4o"
            assert call_kwargs["backend_kwargs"]["max_tokens"] == 8000
            assert call_kwargs["backend_kwargs"]["temperature"] == 0.7

    @patch("rlm.refactor.cli.RLM")
    def test_max_root_tokens_passed_to_rlm(self, mock_rlm):
        """CLI should pass max_root_tokens to RLM constructor, not backend_kwargs."""
        from rlm.refactor.cli import main

        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.response = "Done"
        mock_result.execution_time = 1.0
        mock_instance.completion.return_value = mock_result
        mock_rlm.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, INSTRUCTIONS_FILENAME), "w") as f:
                f.write("Test task")

            main(
                [
                    "--working-dir",
                    tmpdir,
                    "--model",
                    "gpt-4o",
                    "--max-root-tokens",
                    "500000",
                    "--max-sub-tokens",
                    "200000",
                ]
            )

            call_kwargs = mock_rlm.call_args[1]
            assert call_kwargs["max_root_tokens"] == 500000
            assert call_kwargs["max_sub_tokens"] == 200000
            assert "max_root_tokens" not in call_kwargs["backend_kwargs"]
            assert "max_sub_tokens" not in call_kwargs["backend_kwargs"]

    @patch("rlm.refactor.cli.RLM")
    def test_forbidden_params_in_backend_arg_rejected(self, mock_rlm):
        """CLI should reject max_root_tokens in --backend-arg."""
        from rlm.refactor.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, INSTRUCTIONS_FILENAME), "w") as f:
                f.write("Test task")

            with pytest.raises(SystemExit) as exc_info:
                main(
                    [
                        "--working-dir",
                        tmpdir,
                        "--model",
                        "gpt-4o",
                        "--backend-arg",
                        "max_root_tokens=50000",
                    ]
                )
            assert exc_info.value.code == 1

    @patch("rlm.refactor.cli.RLM")
    def test_sub_backend_kwargs_passed(self, mock_rlm):
        """CLI should pass sub-model backend_kwargs separately."""
        from rlm.refactor.cli import main

        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.response = "Done"
        mock_result.execution_time = 1.0
        mock_instance.completion.return_value = mock_result
        mock_rlm.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, INSTRUCTIONS_FILENAME), "w") as f:
                f.write("Test task")

            main(
                [
                    "--working-dir",
                    tmpdir,
                    "--model",
                    "gpt-4o",
                    "--sub-model",
                    "gpt-4o-mini",
                    "--sub-max-tokens",
                    "4000",
                    "--sub-temperature",
                    "0.3",
                ]
            )

            call_kwargs = mock_rlm.call_args[1]
            assert call_kwargs["other_backends"] == ["openai"]
            assert len(call_kwargs["other_backend_kwargs"]) == 1
            sub_kwargs = call_kwargs["other_backend_kwargs"][0]
            assert sub_kwargs["model_name"] == "gpt-4o-mini"
            assert sub_kwargs["max_tokens"] == 4000
            assert sub_kwargs["temperature"] == 0.3


class TestRefactorSystemPrompt:
    """Tests for the refactoring system prompt."""

    def test_prompt_mentions_repos(self):
        assert "/repos" in REFACTOR_SYSTEM_PROMPT

    def test_prompt_mentions_instructions(self):
        assert "rlm-instructions.md" in REFACTOR_SYSTEM_PROMPT

    def test_prompt_mentions_git(self):
        assert "git" in REFACTOR_SYSTEM_PROMPT

    def test_prompt_mentions_llm_query(self):
        assert "llm_query" in REFACTOR_SYSTEM_PROMPT

    def test_prompt_mentions_final_var(self):
        assert "FINAL_VAR" in REFACTOR_SYSTEM_PROMPT

    def test_prompt_mentions_subprocess(self):
        assert "subprocess" in REFACTOR_SYSTEM_PROMPT

    def test_prompt_mentions_repl(self):
        assert "```repl" in REFACTOR_SYSTEM_PROMPT
