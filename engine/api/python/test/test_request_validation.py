"""Validation tests for the private engine request boundary."""

from __future__ import annotations

from copy import deepcopy

import pytest

import pipeline


def _valid_request() -> dict:
    return {
        "parameters": {
            "center": [0.0, 0.0, 0.0],
            "box_size": [30.0, 30.0, 30.0],
            "task": "screen",
            "search_mode": "free",
            "exhaustiveness": 1,
            "randomize": False,
            "mc_steps": 1,
            "opt_steps": 1,
            "refine_steps": 1,
            "num_pose": 1,
            "rmsd_limit": 1.0,
            "energy_range": 5.0,
            "seed": 1,
            "bias": "no",
            "bias_k": 0.1,
            "use_tor_lib": False,
            "energy_decomp": False,
            "constraint_docking": False,
        },
        "runtime": {
            "output_dir": ".",
            "output_prefix": "validation",
            "gpu_device_id": 0,
            "max_gpu_memory": 0,
        },
        "molecules": {"receptor": []},
    }


def test_rejects_missing_top_level_key_before_engine_execution():
    request = _valid_request()
    del request["runtime"]

    with pytest.raises(KeyError, match="runtime"):
        pipeline.run(request)


def test_rejects_invalid_box_triplet_before_engine_execution():
    request = _valid_request()
    request["parameters"]["center"] = [0.0, 0.0]

    with pytest.raises(ValueError, match="center must contain exactly 3 values"):
        pipeline.run(request)


def test_rejects_wrong_scalar_type_before_engine_execution():
    request = _valid_request()
    request["parameters"]["randomize"] = "false"

    with pytest.raises(TypeError, match="randomize must be a boolean"):
        pipeline.run(request)


def test_rejects_non_finite_parameter_before_engine_execution():
    request = _valid_request()
    request["parameters"]["center"][0] = float("nan")

    with pytest.raises(ValueError, match=r"center\[0\] must be finite"):
        pipeline.run(request)


def test_rejects_non_json_numeric_values_before_engine_execution():
    request = deepcopy(_valid_request())
    request["molecules"]["ligand"] = {"value": float("nan")}

    with pytest.raises(ValueError, match="Out of range float values"):
        pipeline.run(request)
