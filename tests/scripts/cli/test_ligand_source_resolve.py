import json
from types import SimpleNamespace

import pytest

from unidock2.cli._resolve import resolve_docking_request, resolve_ligand_inputs, resolve_prepare_ligands_request
from unidock2.config import LIGAND_SOURCE_SDF_FILES, LIGAND_SOURCE_UD2LIG, UnidockConfig
from unidock2.io.ud2lig import UD2LIG_MAGIC, UD2LIG_SPEC_VERSION


def _write_ud2lig_manifest(directory, *, magic=UD2LIG_MAGIC, construct_ff=False):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "magic": magic,
                "spec_version": UD2LIG_SPEC_VERSION,
                "n_ligands": 0,
                "shard_size": 10000,
                "shards": [],
                "prep": {
                    "construct_ff": construct_ff,
                    "template_docking": False,
                    "covalent_ligand": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_sdf(path):
    path.write_text(
        "\n".join(
            [
                "test",
                "     RDKit          2D",
                "",
                "  1  0  0  0  0  0  0  0  0  0999 V2000",
                "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
                "M  END",
                "$$$$",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_resolve_single_sdf(tmp_path):
    ligand = tmp_path / "ligand.sdf"
    _write_sdf(ligand)

    source, files, ud2lig_dir = resolve_ligand_inputs(str(ligand), None, allow_ud2lig=True)

    assert source == LIGAND_SOURCE_SDF_FILES
    assert files == (str(ligand.resolve()),)
    assert ud2lig_dir is None


def test_resolve_sdf_directory_is_sorted(tmp_path):
    ligand_dir = tmp_path / "sdfs"
    ligand_dir.mkdir()
    _write_sdf(ligand_dir / "b.sdf")
    _write_sdf(ligand_dir / "a.sdf")
    (ligand_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    source, files, ud2lig_dir = resolve_ligand_inputs(str(ligand_dir), None, allow_ud2lig=True)

    assert source == LIGAND_SOURCE_SDF_FILES
    assert [path.split("/")[-1] for path in files] == ["a.sdf", "b.sdf"]
    assert ud2lig_dir is None


def test_empty_sdf_directory_is_rejected(tmp_path):
    ligand_dir = tmp_path / "empty"
    ligand_dir.mkdir()

    with pytest.raises(ValueError, match="does not match any supported form"):
        resolve_ligand_inputs(str(ligand_dir), None, allow_ud2lig=True)


def test_resolve_ud2lig_directory(tmp_path):
    library = tmp_path / "lib.ud2lig"
    _write_ud2lig_manifest(library)

    source, files, ud2lig_dir = resolve_ligand_inputs(str(library), None, allow_ud2lig=True)

    assert source == LIGAND_SOURCE_UD2LIG
    assert files == ()
    assert ud2lig_dir == str(library.resolve())


def test_invalid_manifest_is_not_treated_as_sdf_directory(tmp_path):
    library = tmp_path / "lib.ud2lig"
    _write_ud2lig_manifest(library, magic="not-ud2lig")
    _write_sdf(library / "ligand.sdf")

    with pytest.raises(ValueError, match="not a valid UD2LIG manifest"):
        resolve_ligand_inputs(str(library), None, allow_ud2lig=True)


def test_non_sdf_file_is_rejected(tmp_path):
    ligand = tmp_path / "ligand.mol2"
    ligand.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match any supported form"):
        resolve_ligand_inputs(str(ligand), None, allow_ud2lig=True)


def test_missing_path_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not match any supported form"):
        resolve_ligand_inputs(str(tmp_path / "missing.sdf"), None, allow_ud2lig=True)


def test_ud2lig_cannot_combine_with_ligand_batch(tmp_path):
    library = tmp_path / "lib.ud2lig"
    _write_ud2lig_manifest(library)
    batch = tmp_path / "batch.txt"
    batch.write_text("a.sdf\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be combined with -lb"):
        resolve_ligand_inputs(str(library), str(batch), allow_ud2lig=True)


def test_sdf_directory_can_combine_with_ligand_batch(tmp_path):
    ligand_dir = tmp_path / "sdfs"
    ligand_dir.mkdir()
    _write_sdf(ligand_dir / "dir.sdf")
    extra = tmp_path / "extra.sdf"
    _write_sdf(extra)
    batch = tmp_path / "batch.txt"
    batch.write_text(f"{extra}\n", encoding="utf-8")

    source, files, ud2lig_dir = resolve_ligand_inputs(str(ligand_dir), str(batch), allow_ud2lig=True)

    assert source == LIGAND_SOURCE_SDF_FILES
    assert files[-1] == str(extra.resolve())
    assert ud2lig_dir is None


def test_prepare_ligands_rejects_ud2lig_input(tmp_path):
    library = tmp_path / "lib.ud2lig"
    _write_ud2lig_manifest(library)
    args = SimpleNamespace(
        configurations=None,
        ligand=str(library),
        ligand_batch=None,
        construct_ff=None,
        output_ud2lig_dir=str(tmp_path / "out.ud2lig"),
    )

    with pytest.raises(ValueError, match="already a UD2LIG directory"):
        resolve_prepare_ligands_request(args)


def test_docking_rejects_construct_ff_mismatch(tmp_path):
    library = tmp_path / "lib.ud2lig"
    _write_ud2lig_manifest(library, construct_ff=False)
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\n", encoding="utf-8")
    args = SimpleNamespace(
        configurations=None,
        receptor=str(receptor),
        ligand=str(library),
        ligand_batch=None,
        center=None,
        output_docking_pose_sdf_file_name=None,
    )
    config = UnidockConfig().with_overrides(construct_ff=True)

    with pytest.raises(ValueError, match="construct_ff"):
        resolve_docking_request(args, config=config)


def test_docking_cli_uses_output_and_config_long_names():
    import argparse

    from unidock2.cli.docking import CLICommand

    parser = argparse.ArgumentParser()
    CLICommand.add_arguments(parser)
    option_strings = {
        action.dest: action.option_strings
        for action in parser._actions
        if action.option_strings
    }

    assert option_strings["output_docking_pose_sdf_file_name"] == ["-o", "--output"]
    assert option_strings["configurations"] == ["-cf", "--config"]
