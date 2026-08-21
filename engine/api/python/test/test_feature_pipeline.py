"""End-to-end feature tests for the private pybind engine entry point."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import pipeline


ENGINE_ROOT = Path(__file__).resolve().parents[3]
ENGINE_TEST_DATA = ENGINE_ROOT / "test" / "data"


def _load_prepared_input(input_path: Path) -> tuple[list, dict]:
    with input_path.open(encoding="utf-8") as input_file:
        data = json.load(input_file)

    receptor = data.pop("receptor")
    data.pop("score", None)
    return receptor, data


def _run_pipeline(
    *,
    input_path: Path,
    output_dir: Path,
    pipeline_kwargs: dict,
) -> Path:
    receptor, ligands = _load_prepared_input(input_path)
    name_json = input_path.stem

    docking_pipeline = pipeline.DockingPipeline(
        output_dir=str(output_dir),
        name_json=name_json,
        **pipeline_kwargs,
    )
    docking_pipeline.set_receptor(receptor)
    docking_pipeline.add_ligands(ligands)
    docking_pipeline.run()

    output_files = list(output_dir.glob(f"{name_json}_*.json"))
    assert len(output_files) == 1
    return output_files[0]


def _load_first_ligand_poses(output_path: Path) -> list[dict]:
    with output_path.open(encoding="utf-8") as output_file:
        output_data = json.load(output_file)

    assert len(output_data) == 1
    poses = next(iter(output_data.values()))
    assert poses
    return poses


def _parse_sdf_coordinates(sdf_path: Path) -> list[tuple[float, float, float]]:
    with sdf_path.open(encoding="utf-8") as sdf_file:
        lines = sdf_file.readlines()

    atom_count = int(lines[3].split()[0])
    return [tuple(float(value) for value in lines[line_index].split()[:3]) for line_index in range(4, 4 + atom_count)]


def _pose_coordinates(pose: dict) -> list[tuple[float, float, float]]:
    flat_coordinates = pose["coords"]
    return [tuple(flat_coordinates[index : index + 3]) for index in range(0, len(flat_coordinates), 3)]


def _rmsd(
    coordinates: list[tuple[float, float, float]],
    reference: list[tuple[float, float, float]],
) -> float:
    assert len(coordinates) == len(reference)
    squared_distance = sum(
        (x - ref_x) ** 2 + (y - ref_y) ** 2 + (z - ref_z) ** 2
        for (x, y, z), (ref_x, ref_y, ref_z) in zip(coordinates, reference)
    )
    return math.sqrt(squared_distance / len(reference))


def test_5s8i_best_pose_rmsd(tmp_path: Path):
    case_dir = ENGINE_TEST_DATA / "5S8I"
    output_path = _run_pipeline(
        input_path=case_dir / "5S8I_unidock2.json",
        output_dir=tmp_path,
        pipeline_kwargs={
            "center_x": -22.33497980811559,
            "center_y": 13.31094327905649,
            "center_z": 27.36396790165424,
            "size_x": 30.0,
            "size_y": 30.0,
            "size_z": 30.0,
            "task": "screen",
            "search_mode": "free",
            "exhaustiveness": 512,
            "randomize": True,
            "mc_steps": 40,
            "opt_steps": -1,
            "refine_steps": 1,
            "num_pose": 10,
            "rmsd_limit": 1.0,
            "energy_range": 10.0,
            "seed": 121,
            "bias": "no",
            "bias_k": 0.1,
            "constraint_docking": False,
            "use_tor_lib": False,
            "energy_decomp": False,
            "gpu_device_id": 0,
            "max_gpu_mem": 0,
        },
    )

    poses = _load_first_ligand_poses(output_path)
    reference = _parse_sdf_coordinates(case_dir / "5S8I_ligand.sdf")
    best_rmsd = min(_rmsd(_pose_coordinates(pose), reference) for pose in poses)

    assert best_rmsd <= 2.0


def test_position_bias_keeps_anchor_atom_near_target(tmp_path: Path):
    case_dir = ENGINE_TEST_DATA / "bias_hbond"
    output_path = _run_pipeline(
        input_path=case_dir / "input.json",
        output_dir=tmp_path,
        pipeline_kwargs={
            "center_x": 2.0,
            "center_y": 60.0,
            "center_z": 10.0,
            "size_x": 30.0,
            "size_y": 30.0,
            "size_z": 30.0,
            "task": "screen",
            "search_mode": "free",
            "exhaustiveness": 512,
            "randomize": True,
            "mc_steps": 40,
            "opt_steps": -1,
            "refine_steps": 0,
            "num_pose": 10,
            "rmsd_limit": 1.0,
            "energy_range": 10.0,
            "seed": 1234567,
            "bias": "pos",
            "bias_k": 1.0,
            "constraint_docking": False,
            "use_tor_lib": False,
            "energy_decomp": False,
            "gpu_device_id": 0,
            "max_gpu_mem": 0,
        },
    )

    poses = _load_first_ligand_poses(output_path)
    expected_anchor = [1.782, 66.634, 6.337]
    anchor_atom_index = 6

    for pose in poses[:5]:
        coordinate_offset = anchor_atom_index * 3
        actual_anchor = pose["coords"][coordinate_offset : coordinate_offset + 3]
        assert actual_anchor == pytest.approx(expected_anchor, abs=0.2)
