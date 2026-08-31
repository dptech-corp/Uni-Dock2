"""Per-run working directory whose cleanup depends on the run outcome."""

import os
import shutil
import tempfile
from contextlib import contextmanager

from unidock2.io.get_temp_dir_prefix import get_temp_dir_prefix

WORKDIR_ROOT_NAME = "unidock2_temp"

__all__ = ["WORKDIR_ROOT_NAME", "run_workdir", "workdir_root_for"]


def workdir_root_for(output_path):
    """Return the parent directory that holds per-run working directories.

    The root is ``unidock2_temp`` beside the command output, so intermediates
    land on the same disk the user already chose for results.
    """
    output_dir = os.path.dirname(os.path.abspath(output_path))
    return os.path.join(output_dir, WORKDIR_ROOT_NAME)


@contextmanager
def run_workdir(workdir_root, command_name, *, keep=False):
    """Create one working directory per run and report where it is.

    A successful run removes the directory unless ``keep`` is set. A failure
    always leaves it behind so the intermediate files can be inspected.
    """
    workdir_root = os.path.abspath(workdir_root)
    os.makedirs(workdir_root, exist_ok=True)
    workdir = tempfile.mkdtemp(
        prefix=get_temp_dir_prefix(command_name),
        dir=workdir_root,
    )
    print(f"Working directory: {workdir}")

    try:
        yield workdir
    except BaseException:
        print(f"Run failed. Working directory kept for inspection: {workdir}")
        raise

    if keep:
        print(f"Working directory kept: {workdir}")
        return

    shutil.rmtree(workdir, ignore_errors=True)
    try:
        os.rmdir(workdir_root)
    except OSError:
        # Another run still owns a directory here, or the root is not ours to remove.
        pass
