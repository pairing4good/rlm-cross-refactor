"""Tests for the refactoring module: DockerREPL working_dir support and CLI."""

import os
import tempfile
from unittest.mock import MagicMock, patch

from rlm.environments.docker_repl import _build_exec_script
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
