import json

import pytest

from unidock2._engine import build_engine_request
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


def test_engine_request_is_derived_from_config_and_runtime():
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
            "gpu_device_id": 2,
            "max_gpu_memory": 4096,
        },
        "molecules": {"receptor": receptor, **ligands},
    }


def test_engine_request_is_strict_json_serializable():
    receptor, ligands = _minimal_molecules()
    request = build_engine_request(
        UnidockConfig(),
        target_center=(1, 2, 3),
        receptor=receptor,
        ligands=ligands,
    )

    assert json.loads(json.dumps(request, allow_nan=False)) == request


def test_engine_request_rejects_non_serializable_numbers():
    receptor, ligands = _minimal_molecules()
    request = build_engine_request(
        UnidockConfig(),
        target_center=(1, 2, 3),
        receptor=receptor,
        ligands=ligands,
    )
    request["molecules"]["receptor"][0][0] = float("nan")

    with pytest.raises(ValueError, match="Out of range float values"):
        json.dumps(request, allow_nan=False)


def test_engine_request_rejects_reserved_receptor_ligand_key():
    receptor, ligands = _minimal_molecules()
    ligands["receptor"] = []

    with pytest.raises(ValueError, match="reserved 'receptor'"):
        build_engine_request(
            UnidockConfig(),
            target_center=(1, 2, 3),
            receptor=receptor,
            ligands=ligands,
        )
