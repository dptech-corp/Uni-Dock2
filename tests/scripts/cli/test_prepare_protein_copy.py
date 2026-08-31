from types import SimpleNamespace

from unidock2.cli import prepare_protein
from unidock2.unidocktools import unidock_receptor_topology_builder as builder_module


def test_prepare_protein_copies_dms_when_paths_contain_spaces(tmp_path, monkeypatch):
    root_temp_dir = tmp_path / "temporary root"
    root_temp_dir.mkdir()
    output_file = tmp_path / "prepared receptor output.dms"
    request = SimpleNamespace(
        receptor_file_name=str(tmp_path / "input receptor.pdb"),
        workdir_root=str(root_temp_dir),
        receptor_dms_file_name=str(output_file),
        keep_workdir=False,
        config=SimpleNamespace(
            preprocessing=SimpleNamespace(
                preserve_receptor_hydrogen=True,
                covalent_residue_atom_info_list=None,
            )
        ),
    )

    class FakeReceptorBuilder:
        def __init__(self, receptor_file_name, **kwargs):
            del receptor_file_name
            self.receptor_parameterized_dms_file_name = str(kwargs["working_dir_name"] + "/parameterized receptor.dms")

        def generate_receptor_topology(self):
            with open(self.receptor_parameterized_dms_file_name, "wb") as output:
                output.write(b"DMS content")

    monkeypatch.setattr(prepare_protein, "resolve_prepare_protein_request", lambda args: request)
    monkeypatch.setattr(builder_module, "UnidockReceptorTopologyBuilder", FakeReceptorBuilder)

    prepare_protein.CLICommand.run(SimpleNamespace())

    assert output_file.read_bytes() == b"DMS content"
