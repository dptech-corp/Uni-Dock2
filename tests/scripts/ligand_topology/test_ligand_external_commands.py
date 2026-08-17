from pathlib import Path

from unidock2.ligand_topology import utils as ligand_utils


def test_gaff2_tools_use_structured_arguments_and_remove_only_known_temporary_file(tmp_path, monkeypatch):
    working_dir = tmp_path / "ambertools work directory"
    working_dir.mkdir()
    ligand_sdf_file = working_dir / "ligand input.sdf"
    ligand_mol2_file = working_dir / "ligand output.mol2"
    ligand_frcmod_file = working_dir / "ligand output.frcmod"
    temporary_frcmod_file = working_dir / "ANTECHAMBER.FRCMOD"
    temporary_frcmod_file.write_text("temporary\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        ligand_utils,
        "run_external_command",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    ligand_utils._run_ambertools_for_gaff2(
        str(working_dir),
        str(ligand_sdf_file),
        str(ligand_mol2_file),
        str(ligand_frcmod_file),
        "bcc",
        -1,
    )

    assert not temporary_frcmod_file.exists()
    assert calls == [
        (
            [
                "antechamber",
                "-i",
                ligand_sdf_file.name,
                "-fi",
                "sdf",
                "-o",
                ligand_mol2_file.name,
                "-fo",
                "mol2",
                "-at",
                "gaff2",
                "-c",
                "bcc",
                "-nc",
                "-1",
                "-eq",
                "2",
                "-pf",
                "y",
            ],
            {
                "cwd": Path(working_dir),
                "log_file_name": "ligand_temp_antechamber.log",
                "append_log": True,
                "expected_output_file_names": [str(ligand_mol2_file)],
            },
        ),
        (
            [
                "parmchk2",
                "-i",
                ligand_mol2_file.name,
                "-f",
                "mol2",
                "-a",
                "Y",
                "-s",
                "2",
                "-o",
                ligand_frcmod_file.name,
            ],
            {
                "cwd": Path(working_dir),
                "log_file_name": "ligand_temp_parmchk2.log",
                "expected_output_file_names": [str(ligand_frcmod_file)],
            },
        ),
    ]
