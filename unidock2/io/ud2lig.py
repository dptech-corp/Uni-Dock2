"""UD2LIG directory format: reusable preprocessed ligand libraries."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


UD2LIG_MAGIC = "ud2lig"
UD2LIG_SPEC_VERSION = 1
UD2LIG_MANIFEST_NAME = "manifest.json"
DEFAULT_SHARD_SIZE = 10000

LIGAND_KIND_SDF_FILE = "sdf_file"
LIGAND_KIND_SDF_DIR = "sdf_dir"
LIGAND_KIND_UD2LIG = "ud2lig"

_LIGAND_INPUT_HELP = (
    "a single .sdf file, a directory of .sdf files, or a UD2LIG directory "
    f"containing a valid {UD2LIG_MANIFEST_NAME}"
)


def list_sdf_files(directory: str | Path) -> list[str]:
    """Return sorted non-recursive SDF paths in a directory."""
    directory = os.path.abspath(directory)
    names = [
        name
        for name in os.listdir(directory)
        if name.lower().endswith(".sdf") and os.path.isfile(os.path.join(directory, name))
    ]
    return [os.path.join(directory, name) for name in sorted(names)]


def _unmatched_ligand_error(path: str) -> ValueError:
    return ValueError(
        f"Ligand input {path!r} does not match any supported form: {_LIGAND_INPUT_HELP}."
    )


def load_ud2lig_manifest(manifest_file: str | Path) -> dict[str, Any]:
    """Load and validate a UD2LIG manifest. Invalid magic is never an SDF directory."""
    manifest_path = Path(manifest_file)
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict) or manifest.get("magic") != UD2LIG_MAGIC:
        raise ValueError(
            f"{manifest_path} exists but is not a valid UD2LIG manifest "
            f'(expected magic "{UD2LIG_MAGIC}").'
        )
    if manifest.get("spec_version") != UD2LIG_SPEC_VERSION:
        raise ValueError(
            f"{manifest_path} has unsupported spec_version "
            f"{manifest.get('spec_version')!r}; expected {UD2LIG_SPEC_VERSION}."
        )
    if "shards" not in manifest or "prep" not in manifest:
        raise ValueError(f"{manifest_path} is missing required UD2LIG fields.")
    return manifest


def classify_ligand_path(path: str | Path) -> str:
    """Classify -l / Required.ligand as a single SDF, SDF directory, or UD2LIG directory."""
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        raise _unmatched_ligand_error(resolved)

    if os.path.isfile(resolved):
        if resolved.lower().endswith(".sdf"):
            return LIGAND_KIND_SDF_FILE
        raise _unmatched_ligand_error(resolved)

    if not os.path.isdir(resolved):
        raise _unmatched_ligand_error(resolved)

    manifest_path = os.path.join(resolved, UD2LIG_MANIFEST_NAME)
    if os.path.isfile(manifest_path):
        load_ud2lig_manifest(manifest_path)
        return LIGAND_KIND_UD2LIG

    if list_sdf_files(resolved):
        return LIGAND_KIND_SDF_DIR
    raise _unmatched_ligand_error(resolved)


def prep_from_config(config) -> dict[str, bool]:
    """Record the ligand-prep fields stored in a UD2LIG manifest."""
    return {
        "construct_ff": bool(config.preprocessing.construct_ff),
        "template_docking": bool(config.preprocessing.template_docking),
        "covalent_ligand": bool(config.preprocessing.covalent_ligand),
    }


def validate_generic_prep(prep: dict[str, Any], *, context: str) -> None:
    """First version only supports generic (non-template, non-covalent) libraries."""
    if prep.get("template_docking") or prep.get("covalent_ligand"):
        raise ValueError(
            f"{context} uses template or covalent ligand preparation; "
            "the first UD2LIG version only supports generic ligands."
        )


def validate_ud2lig_against_config(manifest: dict[str, Any], config) -> None:
    """Reject UD2LIG reuse when the current config would have prepared ligands differently."""
    prep = manifest["prep"]
    validate_generic_prep(prep, context="UD2LIG library")
    validate_generic_prep(prep_from_config(config), context="Current configuration")

    current = prep_from_config(config)
    mismatches = [
        field_name
        for field_name, value in current.items()
        if prep.get(field_name) != value
    ]
    if mismatches:
        names = ", ".join(mismatches)
        raise ValueError(
            f"UD2LIG library prep fields do not match the current configuration: {names}."
        )


def write_ud2lig(
    output_dir: str | Path,
    ligand_info_dict: dict[str, Any],
    ligand_mol_list,
    prep: dict[str, bool],
    shard_size: int = DEFAULT_SHARD_SIZE,
    overwrite: bool = False,
) -> Path:
    """Write topology shards, chemistry SDFs, and a manifest."""
    from rdkit import Chem

    validate_generic_prep(prep, context="UD2LIG output")
    if shard_size < 1:
        raise ValueError("UD2LIG shard_size must be at least 1")
    if len(ligand_mol_list) == 0:
        raise ValueError("Cannot write an empty UD2LIG directory")
    if len(ligand_mol_list) != len(ligand_info_dict):
        raise ValueError("Ligand topology count does not match chemistry-layer molecule count")

    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        if output_path.is_file():
            raise ValueError(f"UD2LIG output path exists as a file: {output_path}")
        if any(output_path.iterdir()):
            if not overwrite:
                raise ValueError(f"Output UD2LIG directory is not empty: {output_path}")
            shutil.rmtree(output_path)
    shards_dir = output_path / "shards"
    shards_dir.mkdir(parents=True)

    shard_records = []
    for start in range(0, len(ligand_mol_list), shard_size):
        chunk = ligand_mol_list[start : start + shard_size]
        shard_id = f"{start // shard_size:05d}"
        topo: dict[str, Any] = {}
        for molecule in chunk:
            if not molecule.HasProp("ud2_molecule_name"):
                raise ValueError("Chemistry-layer molecule is missing ud2_molecule_name")
            name = molecule.GetProp("ud2_molecule_name")
            if name not in ligand_info_dict:
                raise ValueError(f"No topology entry for molecule {name!r}")
            topo[name] = ligand_info_dict[name]
            if molecule.GetNumAtoms() != len(topo[name]["atoms"]):
                raise ValueError(
                    f"Atom-order invariant failed for {name!r}: "
                    f"{molecule.GetNumAtoms()} chemistry atoms vs "
                    f"{len(topo[name]['atoms'])} topology atoms"
                )

        topo_name = f"{shard_id}.json"
        chem_name = f"{shard_id}.sdf"
        topo_path = shards_dir / topo_name
        chem_path = shards_dir / chem_name
        with topo_path.open("w", encoding="utf-8") as file:
            json.dump(topo, file)
        writer = Chem.SDWriter(str(chem_path))
        writer.SetKekulize(False)
        for molecule in chunk:
            writer.write(molecule)
        writer.close()
        shard_records.append(
            {
                "id": shard_id,
                "n": len(chunk),
                "topo": f"shards/{topo_name}",
                "chem": f"shards/{chem_name}",
            }
        )

    manifest = {
        "magic": UD2LIG_MAGIC,
        "spec_version": UD2LIG_SPEC_VERSION,
        "n_ligands": len(ligand_mol_list),
        "shard_size": shard_size,
        "shards": shard_records,
        "prep": {
            "construct_ff": bool(prep["construct_ff"]),
            "template_docking": bool(prep["template_docking"]),
            "covalent_ligand": bool(prep["covalent_ligand"]),
        },
    }
    with (output_path / UD2LIG_MANIFEST_NAME).open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return output_path


def read_ud2lig(directory: str | Path) -> tuple[dict[str, Any], list, dict[str, Any]]:
    """Load a UD2LIG directory into engine topology and RDKit molecules."""
    from rdkit import Chem

    directory_path = Path(directory).expanduser().resolve()
    manifest = load_ud2lig_manifest(directory_path / UD2LIG_MANIFEST_NAME)
    validate_generic_prep(manifest["prep"], context=str(directory_path))

    ligand_info_dict: dict[str, Any] = {}
    ligand_mol_list = []
    for shard in manifest["shards"]:
        topo_path = directory_path / shard["topo"]
        chem_path = directory_path / shard["chem"]
        with topo_path.open(encoding="utf-8") as file:
            shard_topo = json.load(file)
        molecules = list(Chem.SDMolSupplier(str(chem_path), removeHs=False))
        if len(molecules) != shard["n"] or len(shard_topo) != shard["n"]:
            raise ValueError(
                f"UD2LIG shard {shard['id']!r} count mismatch: "
                f"manifest n={shard['n']}, chemistry={len(molecules)}, topology={len(shard_topo)}"
            )
        for molecule in molecules:
            if molecule is None:
                raise ValueError(f"UD2LIG chemistry file {chem_path} contains an unreadable molecule")
            if not molecule.HasProp("ud2_molecule_name"):
                raise ValueError(f"UD2LIG chemistry file {chem_path} is missing ud2_molecule_name")
            name = molecule.GetProp("ud2_molecule_name")
            if name not in shard_topo:
                raise ValueError(f"UD2LIG chemistry molecule {name!r} has no topology in {topo_path}")
            if molecule.GetNumAtoms() != len(shard_topo[name]["atoms"]):
                raise ValueError(
                    f"Atom-order invariant failed for {name!r} in shard {shard['id']!r}"
                )
            ligand_info_dict[name] = shard_topo[name]
            ligand_mol_list.append(molecule)

    if len(ligand_mol_list) != manifest["n_ligands"]:
        raise ValueError(
            f"UD2LIG {directory_path} expected {manifest['n_ligands']} ligands, "
            f"loaded {len(ligand_mol_list)}"
        )
    return ligand_info_dict, ligand_mol_list, manifest
