from unidock2.unidocktools import receptor_topology_preparation as preparation_module
from unidock2.unidocktools import unidock_receptor_topology_builder as builder_module


def test_receptor_builder_passes_structured_arguments_to_fepfixer_and_utop(tmp_path, monkeypatch):
    working_dir = tmp_path / "receptor work directory"
    working_dir.mkdir()
    receptor_file = tmp_path / "input receptor.pdb"
    receptor_file.write_text("MODEL\nEND\n", encoding="utf-8")
    executables = {
        "fepfixer": "/opt/test tools/fepfixer",
        "utop": "/opt/test tools/utop",
    }
    calls = []

    monkeypatch.setattr(builder_module, "which", executables.get)
    monkeypatch.setattr(
        builder_module,
        "run_external_command",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    builder = builder_module.UnidockReceptorTopologyBuilder(
        str(receptor_file),
        prepared_hydrogen=True,
        working_dir_name=str(working_dir),
    )
    builder.run_protein_preparation()

    assert calls == [
        (
            [
                executables["fepfixer"],
                "-i",
                str(receptor_file),
                "-o",
                "receptor_structure.dms",
                "--custom-protonation-states",
            ],
            {
                "cwd": str(working_dir),
                "log_file_name": "fepfixer.log",
                "expected_output_file_names": [str(working_dir / "receptor_structure.dms")],
            },
        ),
        (
            [
                executables["utop"],
                "prm",
                "-i",
                "receptor_structure.dms",
                "-o",
                "receptor_parameterized.dms",
            ],
            {
                "cwd": str(working_dir),
                "log_file_name": "utop.log",
                "expected_output_file_names": [str(working_dir / "receptor_parameterized.dms")],
            },
        ),
    ]


def test_tleap_uses_working_directory_log_and_declared_outputs(tmp_path, monkeypatch):
    working_dir = tmp_path / "tleap work directory"
    working_dir.mkdir()
    calls = []
    monkeypatch.setattr(
        preparation_module,
        "run_external_command",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    preparation = preparation_module.ReceptorTopologyPreparation(
        str(tmp_path / "input receptor.pdb"),
        str(working_dir),
    )
    preparation._run_tleap()

    assert "loadPdb receptor_final.pdb" in (working_dir / "tleap.in").read_text(encoding="utf-8")
    assert calls == [
        (
            ["tleap", "-f", "tleap.in"],
            {
                "cwd": str(working_dir),
                "log_file_name": "tleap.log",
                "append_log": True,
                "expected_output_file_names": [
                    str(working_dir / "receptor.prmtop"),
                    str(working_dir / "receptor.inpcrd"),
                ],
            },
        )
    ]
