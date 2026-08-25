import json

import pytest

from unidock2._engine import (
    build_engine_request,
    dump_engine_request,
    load_engine_request,
)
from unidock2.config import UnidockConfig


def _minimal_molecules():
    receptor = [[0.0, 0.0, 0.0, 2, 0, 0.0]]
    ligands = {
        "MOL_0": {
            "atoms": [[1.0, 0.0, 0.0, 2, 0, 0.0, [], []]],
            "torsions": [],
            "root_atoms": [0],
        }
    }
    return receptor, ligands


def test_engine_request_is_derived_from_config_and_runtime(tmp_path):
    config = UnidockConfig().with_overrides(
        box_size=[10, 20, 30],
        task="score",
        search_mode="free",
        exhaustiveness=64,
        randomize=False,
        mc_steps=12,
        opt_steps=13,
        refine_steps=14,
        num_pose=4,
        rmsd_limit=1.5,
        energy_range=6.0,
        seed=42,
        bias="pos",
        bias_k=0.25,
        use_tor_lib=True,
        energy_decomp=True,
        template_docking=True,
        gpu_device_id=2,
        max_gpu_memory=4096,
    )
    receptor, ligands = _minimal_molecules()

    request = build_engine_request(
        config,
        target_center=(1, 2, 3),
        output_dir=tmp_path / "output",
        output_prefix="batch",
        receptor=receptor,
        ligands=ligands,
    )

    assert request == {
        "parameters": {
            "center": [1.0, 2.0, 3.0],
            "box_size": [10.0, 20.0, 30.0],
            "task": "score",
            "search_mode": "free",
            "exhaustiveness": 64,
            "randomize": False,
            "mc_steps": 12,
            "opt_steps": 13,
            "refine_steps": 14,
            "num_pose": 4,
            "rmsd_limit": 1.5,
            "energy_range": 6.0,
            "seed": 42,
            "bias": "pos",
            "bias_k": 0.25,
            "use_tor_lib": True,
            "energy_decomp": True,
            "constraint_docking": True,
        },
        "runtime": {
            "output_dir": str(tmp_path / "output"),
            "output_prefix": "batch",
            "gpu_device_id": 2,
            "max_gpu_memory": 4096,
        },
        "molecules": {"receptor": receptor, **ligands},
    }


def test_engine_request_round_trips_through_strict_json(tmp_path):
    receptor, ligands = _minimal_molecules()
    request = build_engine_request(
        UnidockConfig(),
        target_center=(1, 2, 3),
        output_dir=tmp_path / "output",
        receptor=receptor,
        ligands=ligands,
    )

    checkpoint_path = dump_engine_request(request, tmp_path / "request.json")

    assert load_engine_request(checkpoint_path) == request
    assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == request


def test_engine_request_rejects_non_serializable_numbers(tmp_path):
    receptor, ligands = _minimal_molecules()
    request = build_engine_request(
        UnidockConfig(),
        target_center=(1, 2, 3),
        output_dir=tmp_path / "output",
        receptor=receptor,
        ligands=ligands,
    )
    request["molecules"]["receptor"][0][0] = float("nan")

    with pytest.raises(ValueError, match="Out of range float values"):
        dump_engine_request(request, tmp_path / "request.json")


def test_engine_request_rejects_reserved_receptor_ligand_key(tmp_path):
    receptor, ligands = _minimal_molecules()
    ligands["receptor"] = []

    with pytest.raises(ValueError, match="reserved 'receptor'"):
        build_engine_request(
            UnidockConfig(),
            target_center=(1, 2, 3),
            output_dir=tmp_path / "output",
            receptor=receptor,
            ligands=ligands,
        )
