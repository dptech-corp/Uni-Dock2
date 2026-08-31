import os

import pytest

from unidock2.io.workdir import WORKDIR_ROOT_NAME, run_workdir, workdir_root_for


def test_root_sits_beside_the_command_output(tmp_path):
    output_file = tmp_path / "results" / "poses.sdf"

    root = workdir_root_for(str(output_file))

    assert root == str(tmp_path / "results" / WORKDIR_ROOT_NAME)


def test_relative_output_resolves_against_the_process_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    root = workdir_root_for("results/poses.sdf")

    assert root == str(tmp_path / "results" / WORKDIR_ROOT_NAME)


def test_successful_run_removes_the_working_directory(tmp_path):
    root = tmp_path / WORKDIR_ROOT_NAME

    with run_workdir(str(root), "docking") as workdir:
        assert os.path.isdir(workdir)
        with open(os.path.join(workdir, "intermediate.sdf"), "w", encoding="utf-8") as file:
            file.write("x")
        escaped_workdir = workdir

    assert not os.path.exists(escaped_workdir)


def test_successful_run_leaves_no_empty_root_behind(tmp_path):
    root = tmp_path / WORKDIR_ROOT_NAME

    with run_workdir(str(root), "docking"):
        pass

    assert not os.path.exists(root)


def test_root_survives_while_another_run_still_owns_a_directory(tmp_path):
    root = tmp_path / WORKDIR_ROOT_NAME

    with run_workdir(str(root), "docking", keep=True) as kept_workdir:
        pass
    with run_workdir(str(root), "docking"):
        pass

    assert os.path.isdir(kept_workdir)
    assert os.path.isdir(root)


def test_failed_run_keeps_the_working_directory_for_inspection(tmp_path):
    root = tmp_path / WORKDIR_ROOT_NAME

    with pytest.raises(RuntimeError, match="ligand preparation failed"):
        with run_workdir(str(root), "docking") as workdir:
            escaped_workdir = workdir
            with open(os.path.join(workdir, "tleap.in"), "w", encoding="utf-8") as file:
                file.write("x")
            raise RuntimeError("ligand preparation failed")

    assert os.path.isfile(os.path.join(escaped_workdir, "tleap.in"))


def test_keep_workdir_retains_a_successful_run(tmp_path):
    root = tmp_path / WORKDIR_ROOT_NAME

    with run_workdir(str(root), "docking", keep=True) as workdir:
        escaped_workdir = workdir

    assert os.path.isdir(escaped_workdir)


def test_each_run_gets_its_own_directory_under_the_same_root(tmp_path):
    root = tmp_path / WORKDIR_ROOT_NAME
    workdir_names = set()

    for _ in range(3):
        with run_workdir(str(root), "docking", keep=True) as workdir:
            workdir_names.add(workdir)
            assert os.path.dirname(workdir) == str(root)

    assert len(workdir_names) == 3
