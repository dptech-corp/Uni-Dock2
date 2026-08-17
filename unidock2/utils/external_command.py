"""Safe boundary for command-line tools used by Uni-Dock2."""

from collections.abc import Sequence
import os
from pathlib import Path
import shlex
import shutil
import subprocess


PathArgument = str | os.PathLike[str]


class ExternalCommandError(RuntimeError):
    """Report an external command failure with its diagnostic context."""

    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str],
        returncode: int | None,
        log_file_name: Path | None = None,
        missing_output_file_names: Sequence[Path] = (),
    ) -> None:
        super().__init__(message)
        self.command = tuple(command)
        self.returncode = returncode
        self.log_file_name = log_file_name
        self.missing_output_file_names = tuple(missing_output_file_names)


def _path_from_cwd(file_name: PathArgument, cwd: Path | None) -> Path:
    path = Path(file_name)
    if path.is_absolute() or cwd is None:
        return path
    return cwd / path


def run_external_command(
    command: Sequence[PathArgument],
    *,
    cwd: PathArgument | None = None,
    log_file_name: PathArgument | None = None,
    append_log: bool = False,
    expected_output_file_names: Sequence[PathArgument] = (),
) -> subprocess.CompletedProcess:
    """Run a command without a shell and validate its declared output files.

    Relative log and output paths are interpreted from ``cwd``. When a log is
    requested, stderr is merged into stdout so failures have one diagnostic
    location.
    """

    if not command:
        raise ValueError("External command must contain an executable")

    command_args = [os.fspath(argument) for argument in command]
    executable = shutil.which(command_args[0])
    if executable is None:
        raise FileNotFoundError(f"Required external program was not found or is not executable: {command_args[0]}")
    command_args[0] = executable

    cwd_path = Path(cwd) if cwd is not None else None
    log_path = _path_from_cwd(log_file_name, cwd_path) if log_file_name is not None else None

    try:
        if log_path is None:
            result = subprocess.run(
                command_args,
                check=True,
                cwd=cwd_path,
                shell=False,
            )
        else:
            log_mode = "a" if append_log else "w"
            with log_path.open(log_mode, encoding="utf-8") as log_file:
                result = subprocess.run(
                    command_args,
                    check=True,
                    cwd=cwd_path,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    shell=False,
                )
    except subprocess.CalledProcessError as error:
        message = f"External command failed with exit code {error.returncode}: {shlex.join(command_args)}"
        if log_path is not None:
            message += f"; see log: {log_path}"
        raise ExternalCommandError(
            message,
            command=command_args,
            returncode=error.returncode,
            log_file_name=log_path,
        ) from error

    expected_paths = tuple(_path_from_cwd(file_name, cwd_path) for file_name in expected_output_file_names)
    missing_output_paths = tuple(path for path in expected_paths if not path.is_file() or path.stat().st_size == 0)
    if missing_output_paths:
        missing_names = ", ".join(str(path) for path in missing_output_paths)
        message = (
            f"External command completed without producing non-empty output file(s): {missing_names}; "
            f"command: {shlex.join(command_args)}"
        )
        if log_path is not None:
            message += f"; see log: {log_path}"
        raise ExternalCommandError(
            message,
            command=command_args,
            returncode=result.returncode,
            log_file_name=log_path,
            missing_output_file_names=missing_output_paths,
        )

    return result
