"""Build and serialize the private, versioned native-engine request."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, TypedDict, cast

from unidock2.config import UnidockConfig


ENGINE_REQUEST_SCHEMA_VERSION = 1
DEFAULT_ENGINE_OUTPUT_PREFIX = "from_python_obj"


class EngineRequest(TypedDict):
    """JSON-compatible data accepted by the private native engine binding."""

    schema_version: int
    parameters: dict[str, Any]
    runtime: dict[str, Any]
    molecules: dict[str, Any]


def build_engine_request(
    config: UnidockConfig,
    *,
    target_center,
    output_dir: str | Path,
    receptor: list,
    ligands: Mapping[str, Any],
    output_prefix: str = DEFAULT_ENGINE_OUTPUT_PREFIX,
) -> EngineRequest:
    """Translate validated public configuration and prepared topology to the native contract."""
    if not isinstance(config, UnidockConfig):
        config = UnidockConfig.model_validate(config)

    center = [float(value) for value in target_center]
    if len(center) != 3:
        raise ValueError("Engine target center requires 3 elements")

    ligand_data = dict(ligands)
    if "receptor" in ligand_data:
        raise ValueError("Ligand topology must not contain the reserved 'receptor' key")

    advanced = config.advanced.model_dump()
    parameters = {
        "center": center,
        "box_size": list(config.settings.box_size),
        "task": config.settings.task,
        "search_mode": config.settings.search_mode,
        **advanced,
        "constraint_docking": (config.preprocessing.template_docking or config.preprocessing.covalent_ligand),
    }
    runtime = {
        "output_dir": str(output_dir),
        "output_prefix": str(output_prefix),
        "gpu_device_id": config.hardware.gpu_device_id,
        "max_gpu_memory": config.hardware.max_gpu_memory,
    }
    molecules = {"receptor": receptor, **ligand_data}

    return {
        "schema_version": ENGINE_REQUEST_SCHEMA_VERSION,
        "parameters": parameters,
        "runtime": runtime,
        "molecules": molecules,
    }


def dump_engine_request(request: EngineRequest, output_file: str | Path) -> Path:
    """Write a request as strict JSON and return its absolute path."""
    output_path = Path(output_file).expanduser().resolve()
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(request, file, allow_nan=False)
    return output_path


def load_engine_request(input_file: str | Path) -> EngineRequest:
    """Load a JSON request; native schema validation occurs when it is run."""
    input_path = Path(input_file).expanduser().resolve()
    with input_path.open(encoding="utf-8") as file:
        request = json.load(file)

    if not isinstance(request, dict):
        raise ValueError("Engine request must be a JSON object")
    return cast(EngineRequest, request)
