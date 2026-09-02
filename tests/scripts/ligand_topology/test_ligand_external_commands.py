from pathlib import Path

from unidock2.force_field import ligand_gaff2
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
        ligand_gaff2,
        "run_external_command",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    ligand_gaff2._run_ambertools_for_gaff2(
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


def test_disabled_ligand_force_field_data_preserves_engine_placeholders():
    class Molecule:
        def GetNumAtoms(self):
            return 3

    atom_types, partial_charges, torsion_parameters = ligand_gaff2.get_ligand_force_field_data(Molecule(), False, ".")

    assert atom_types == ["c", "c", "c"]
    assert partial_charges == [0.0, 0.0, 0.0]
    assert torsion_parameters == {}
    assert ligand_gaff2.encode_atom_force_field(atom_types[0], partial_charges[0]) == (0, 0.0)


def test_legacy_ligand_utils_reexport_public_force_field_functions():
    assert (
        ligand_utils.convert_v3000_mol_to_v2000_sdf
        is ligand_gaff2.convert_v3000_mol_to_v2000_sdf
    )
    assert (
        ligand_utils.record_gaff2_atom_types_and_parameters
        is ligand_gaff2.record_gaff2_atom_types_and_parameters
    )


def test_torsion_force_field_parameters_preserve_reverse_lookup_and_field_order():
    parameter = {
        "barrier_factor": 2,
        "barrier_height": 1.25,
        "periodicity": 3,
        "phase": 180.0,
    }
    torsion_parameter_nested_dict = {("c", "c3", "n", "o"): [parameter]}

    result = ligand_gaff2.get_torsion_force_field_parameters(
        [0, 1, 2, 3],
        ["o", "n", "c3", "c"],
        torsion_parameter_nested_dict,
        True,
    )

    assert result == [[2, 1.25, 3, 180.0]]
    assert (
        ligand_gaff2.get_torsion_force_field_parameters(
            [0, 1, 2, 3],
            ["o", "n", "c3", "c"],
            torsion_parameter_nested_dict,
            False,
        )
        == []
    )
