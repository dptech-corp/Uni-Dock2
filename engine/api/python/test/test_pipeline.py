"""Smoke test for the private pybind engine entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pipeline


TEST_DATA = Path(__file__).resolve().parent / "data" / "1W1P"


def test_pipeline_smoke(tmp_path: Path):
    input_path = TEST_DATA / "1W1P_unidock2.json"
    with input_path.open(encoding="utf-8") as input_file:
        input_data = json.load(input_file)

    receptor = input_data.pop("receptor")
    input_data.pop("score", None)

    pipeline.run(
        {
            "schema_version": pipeline.ENGINE_REQUEST_SCHEMA_VERSION,
            "parameters": {
                "center": [43.20550987534587, 75.61079763066026, 51.93665163136053],
                "box_size": [30.0, 30.0, 30.0],
                "task": "screen",
                "search_mode": "free",
                "exhaustiveness": 512,
                "randomize": True,
                "mc_steps": 40,
                "opt_steps": -1,
                "refine_steps": 0,
                "num_pose": 1,
                "rmsd_limit": 1.0,
                "energy_range": 10.0,
                "seed": 121,
                "bias": "no",
                "bias_k": 0.1,
                "constraint_docking": False,
                "use_tor_lib": False,
                "energy_decomp": False,
            },
            "runtime": {
                "output_dir": str(tmp_path),
                "output_prefix": input_path.stem,
                "gpu_device_id": 0,
                "max_gpu_memory": 0,
            },
            "molecules": {"receptor": receptor, **input_data},
        }
    )

    output_files = list(tmp_path.glob(f"{input_path.stem}_*.json"))
    assert len(output_files) == 1

    with output_files[0].open(encoding="utf-8") as output_file:
        output_data = json.load(output_file)

    assert len(output_data) == 1
    assert next(iter(output_data.values()))
