import sys
import types
from types import SimpleNamespace

import os

from rdkit import Chem

from context import TEST_DATA_DIR
from unidock2.cli import prepare_ligands
from unidock2.config import UnidockConfig
from unidock2.io.ud2lig import prep_from_config, write_ud2lig
from unidock2.unidocktools.unidock_protocol_runner import UnidockProtocolRunner


def _named_mol(name):
    ligand_file = os.path.join(TEST_DATA_DIR, "ligand_topology", "test_vina_atom_type_1.sdf")
    molecule = Chem.SDMolSupplier(ligand_file, removeHs=False)[0]
    molecule.SetProp("ud2_molecule_name", name)
    return molecule


def _topology(molecule):
    conformer = molecule.GetConformer()
    atoms = []
    for atom_idx in range(molecule.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_idx)
        atoms.append([position.x, position.y, position.z, 2, 0, 0.0, [], []])
    return {
        "atoms": atoms,
        "torsions": [],
        "root_atoms": [0],
        "fragment_atom_idx": [[0]],
    }


def test_prepare_ligands_writes_ud2lig(tmp_path, monkeypatch):
    mol_0 = _named_mol("MOL_0")

    class FakeBuilder:
        def __init__(self, *args, **kwargs):
            self.ligand_mol_list = [mol_0]
            self.summary_ligand_info_dict = {"MOL_0": _topology(mol_0)}

        def generate_batch_ligand_topology(self):
            return None

        def get_summary_ligand_info_dict(self):
            return None

    request = SimpleNamespace(
        ligand_sdf_file_name_list=(str(tmp_path / "ligand.sdf"),),
        output_ud2lig_dir=str(tmp_path / "lib.ud2lig"),
        workdir_root=str(tmp_path / "tmp"),
        keep_workdir=False,
        config=UnidockConfig(),
    )
    monkeypatch.setattr(prepare_ligands, "resolve_prepare_ligands_request", lambda args: request)
    monkeypatch.setattr(
        "unidock2.unidocktools.unidock_ligand_topology_builder.UnidockLigandTopologyBuilder",
        FakeBuilder,
    )

    prepare_ligands.CLICommand.run(SimpleNamespace())

    assert (tmp_path / "lib.ud2lig" / "manifest.json").is_file()
    assert (tmp_path / "lib.ud2lig" / "shards" / "00000.sdf").is_file()


def test_runner_uses_ud2lig_without_building_ligand_topology(tmp_path, monkeypatch):
    mol_0 = _named_mol("MOL_0")
    library = tmp_path / "lib.ud2lig"
    write_ud2lig(
        library,
        {"MOL_0": _topology(mol_0)},
        [mol_0],
        prep_from_config(UnidockConfig()),
    )

    class FailLigandBuilder:
        def __init__(self, *args, **kwargs):
            raise AssertionError("ligand topology should not be generated for UD2LIG")

    captured = {}

    def fake_run(request):
        captured["ligands"] = request["molecules"]
        n_coords = mol_0.GetNumAtoms() * 3
        return {
            "MOL_0": [
                {
                    "energy": [0.0] * 7,
                    "coords": [0.0] * n_coords,
                    "dihedrals": [],
                }
            ]
        }

    fake_pipeline = types.ModuleType("unidock2._engine.pipeline")
    fake_pipeline.run = fake_run
    monkeypatch.setitem(sys.modules, "unidock2._engine.pipeline", fake_pipeline)
    monkeypatch.setattr("unidock2._engine.pipeline", fake_pipeline, raising=False)

    monkeypatch.setattr(
        "unidock2.unidocktools.unidock_ligand_topology_builder.UnidockLigandTopologyBuilder",
        FailLigandBuilder,
    )

    class FakePoseWriter:
        def __init__(self, ligand_mol_list, *args, **kwargs):
            captured["mol_names"] = [mol.GetProp("ud2_molecule_name") for mol in ligand_mol_list]

        def generate_docking_pose_sdf(self):
            return None

    monkeypatch.setattr(
        "unidock2.unidocktools.unidock_ligand_pose_writer.UnidockLigandPoseWriter",
        FakePoseWriter,
    )

    receptor = tmp_path / "receptor.json"
    receptor.write_text('{"receptor": [[0.0, 0.0, 0.0, 2, 0, 0.0]]}', encoding="utf-8")
    pose_sdf = tmp_path / "from_ud2lig.sdf"
    runner = UnidockProtocolRunner.from_config(
        str(receptor),
        [],
        (1.0, 2.0, 3.0),
        working_dir_name=str(tmp_path / "work"),
        docking_pose_sdf_file_name=str(pose_sdf),
        ud2lig_dir=str(library),
    )
    runner.run_unidock_protocol()

    assert "MOL_0" in captured["ligands"]
    assert captured["mol_names"] == ["MOL_0"]
    assert not (tmp_path / "from_ud2lig.ud2lig").exists()


def _install_fake_pipeline(monkeypatch, mol_0, captured=None):
    def fake_run(request):
        if captured is not None:
            captured["ligands"] = request["molecules"]
        n_coords = mol_0.GetNumAtoms() * 3
        return {
            "MOL_0": [
                {
                    "energy": [0.0] * 7,
                    "coords": [0.0] * n_coords,
                    "dihedrals": [],
                }
            ]
        }

    fake_pipeline = types.ModuleType("unidock2._engine.pipeline")
    fake_pipeline.run = fake_run
    monkeypatch.setitem(sys.modules, "unidock2._engine.pipeline", fake_pipeline)
    monkeypatch.setattr("unidock2._engine.pipeline", fake_pipeline, raising=False)
    return fake_pipeline


def test_runner_writes_ud2lig_next_to_pose_by_default(tmp_path, monkeypatch):
    mol_0 = _named_mol("MOL_0")

    class FakeBuilder:
        def __init__(self, *args, **kwargs):
            self.ligand_mol_list = [mol_0]
            self.summary_ligand_info_dict = {"MOL_0": _topology(mol_0)}

        def generate_batch_ligand_topology(self):
            return None

        def get_summary_ligand_info_dict(self):
            return None

    _install_fake_pipeline(monkeypatch, mol_0)
    monkeypatch.setattr(
        "unidock2.unidocktools.unidock_ligand_topology_builder.UnidockLigandTopologyBuilder",
        FakeBuilder,
    )

    receptor = tmp_path / "receptor.json"
    receptor.write_text('{"receptor": [[0.0, 0.0, 0.0, 2, 0, 0.0]]}', encoding="utf-8")
    pose_sdf = tmp_path / "poses.sdf"
    runner = UnidockProtocolRunner.from_config(
        str(receptor),
        [str(tmp_path / "ligand.sdf")],
        (1.0, 2.0, 3.0),
        working_dir_name=str(tmp_path / "work"),
        docking_pose_sdf_file_name=str(pose_sdf),
    )
    runner.run_unidock_protocol()

    library = tmp_path / "poses.ud2lig"
    assert (library / "manifest.json").is_file()
    assert pose_sdf.is_file()


def test_runner_skips_ud2lig_dump_when_checkpoint_disabled(tmp_path, monkeypatch):
    mol_0 = _named_mol("MOL_0")

    class FakeBuilder:
        def __init__(self, *args, **kwargs):
            self.ligand_mol_list = [mol_0]
            self.summary_ligand_info_dict = {"MOL_0": _topology(mol_0)}

        def generate_batch_ligand_topology(self):
            return None

        def get_summary_ligand_info_dict(self):
            return None

    _install_fake_pipeline(monkeypatch, mol_0)
    monkeypatch.setattr(
        "unidock2.unidocktools.unidock_ligand_topology_builder.UnidockLigandTopologyBuilder",
        FakeBuilder,
    )

    receptor = tmp_path / "receptor.json"
    receptor.write_text('{"receptor": [[0.0, 0.0, 0.0, 2, 0, 0.0]]}', encoding="utf-8")
    pose_sdf = tmp_path / "poses.sdf"
    runner = UnidockProtocolRunner.from_config(
        str(receptor),
        [str(tmp_path / "ligand.sdf")],
        (1.0, 2.0, 3.0),
        config=UnidockConfig().with_overrides(engine_checkpoint=False),
        working_dir_name=str(tmp_path / "work"),
        docking_pose_sdf_file_name=str(pose_sdf),
    )
    runner.run_unidock_protocol()

    assert not (tmp_path / "poses.ud2lig").exists()
