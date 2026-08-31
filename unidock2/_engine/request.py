"""Build the private native-engine request."""

from __future__ import annotations

from typing import Any, Mapping, TypedDict

from unidock2.config import UnidockConfig


class EngineRequest(TypedDict):
    """JSON-compatible data accepted by the private native engine binding."""

    parameters: dict[str, Any]
    runtime: dict[str, Any]
    molecules: dict[str, Any]


def build_engine_request(
    config: UnidockConfig,
    *,
    target_center,
    receptor: list,
    ligands: Mapping[str, Any],
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
        "gpu_device_id": config.hardware.gpu_device_id,
        "max_gpu_memory": config.hardware.max_gpu_memory,
    }
    molecules = {"receptor": receptor, **ligand_data}

    return {
        "parameters": parameters,
        "runtime": runtime,
        "molecules": molecules,
    }
