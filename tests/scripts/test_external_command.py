import stat
import subprocess

import pytest

from unidock2.utils.external_command import (
    ExternalCommandError,
    run_external_command,
)


def _write_executable(path, body):
    path.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_run_external_command_preserves_paths_arguments_and_logs(tmp_path):
    executable = tmp_path / "fake external tool"
    working_dir = tmp_path / "working directory"
    working_dir.mkdir()
    _write_executable(
        executable,
        """printf '%s\\n' "$@" > received_args.txt
printf '%s\\n' "$PWD" > received_cwd.txt
printf 'result\\n' > 'output file.dat'
printf 'stdout line\\n'
printf 'stderr line\\n' >&2
""",
    )
    log_file = working_dir / "tool output.log"
    log_file.write_text("previous run\n", encoding="utf-8")

    result = run_external_command(
        [executable, "argument with spaces", "literal; touch injected"],
        cwd=working_dir,
        log_file_name="tool output.log",
        append_log=True,
        expected_output_file_names=["output file.dat"],
    )

    assert result.returncode == 0
    assert (working_dir / "received_args.txt").read_text(encoding="utf-8").splitlines() == [
        "argument with spaces",
        "literal; touch injected",
    ]
    assert (working_dir / "received_cwd.txt").read_text(encoding="utf-8").strip() == str(working_dir)
    assert not (working_dir / "injected").exists()
    assert log_file.read_text(encoding="utf-8") == "previous run\nstdout line\nstderr line\n"


def test_run_external_command_reports_nonzero_exit_with_log_path(tmp_path):
    executable = tmp_path / "failing tool"
    log_file = tmp_path / "failure.log"
    _write_executable(executable, "printf 'failure details\\n' >&2\nexit 7\n")

    with pytest.raises(ExternalCommandError, match="exit code 7") as error:
        run_external_command(
            [executable],
            cwd=tmp_path,
            log_file_name=log_file,
        )

    assert error.value.returncode == 7
    assert error.value.log_file_name == log_file
    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)
    assert "failure details" in log_file.read_text(encoding="utf-8")


def test_run_external_command_reports_missing_executable(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")

    with pytest.raises(FileNotFoundError, match="missing-test-program"):
        run_external_command(["missing-test-program"], cwd=tmp_path)


def test_run_external_command_rejects_missing_or_empty_outputs(tmp_path):
    executable = tmp_path / "successful tool"
    _write_executable(executable, "exit 0\n")

    with pytest.raises(ExternalCommandError, match="non-empty output") as error:
        run_external_command(
            [executable],
            cwd=tmp_path,
            expected_output_file_names=["expected output.dat"],
        )

    assert error.value.returncode == 0
    assert error.value.missing_output_file_names == (tmp_path / "expected output.dat",)
