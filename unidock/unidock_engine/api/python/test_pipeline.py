"""
@author Congcong Liu
@date 2026/1/21
@brief pytest tests for pipeline module
"""
# === Setup sys.path FIRST (before any project imports) ===
import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
example_dir = project_root / "examples" / "1W1P"
build_path = project_root / "cmake-build-release" / "api" / "python"

# Add build directory to sys.path (for pipeline.so)
if str(build_path) not in sys.path:
    sys.path.insert(0, str(build_path))
# Also check PYTHONPATH from CTest
for p in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    if p and p not in sys.path:
        sys.path.insert(0, p)

# === Now import project modules ===
import json
import yaml
import pipeline


def load_config(yaml_path: Path) -> dict:
    """Load configuration from ud2.yaml"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def load_molecules(json_path: Path) -> tuple[list, dict]:
    """
    Load receptor and ligands from JSON file.
    Returns:
        receptor_info: list of atom info [x, y, z, vina_type, ff_type, charge]
        ligands_info: dict of ligand_name -> {atoms, torsions, root_atoms, ...}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # The JSON structure has:
    # - "receptor": [[x,y,z,type,...], ...] or the receptor is at "receptor" key
    # - "score": [...] (optional)
    # - "ligand_name": {atoms, torsions, root_atoms, ...}
    
    receptor_info = data.get("receptor", [])
    
    # Everything else except "receptor" and "score" is a ligand
    ligands_info = {k: v for k, v in data.items() if k not in ["receptor", "score"]}
    
    return receptor_info, ligands_info


def run_docking(example_dir: Path):
    """Run docking using configuration from example directory"""
    
    # Load config
    yaml_path = example_dir / "ud2.yaml"
    config = load_config(yaml_path)
    
    # Extract settings
    settings = config.get("Settings", {})
    advanced = config.get("Advanced", {})
    hardware = config.get("Hardware", {})
    outputs = config.get("Outputs", {})
    inputs = config.get("Inputs", {})
    
    # Load molecules from JSON
    json_rel_path = inputs.get("json", "")
    json_path = example_dir / json_rel_path
    print(f"Loading molecules from: {json_path}")
    receptor_info, ligands_info = load_molecules(json_path)

    output_dir = os.path.join(str(example_dir), "res2_py")

    print(f"Receptor atoms: {len(receptor_info)}")
    print(f"Ligands count: {len(ligands_info)}")
    print(f"Ligand names: {list(ligands_info.keys())[:5]}...")  # Show first 5

    # Create DockingPipeline
    print("\nCreating DockingPipeline...")
    docking_pipeline = pipeline.DockingPipeline(
        output_dir=output_dir,
        center_x=settings.get("center_x", 0.0),
        center_y=settings.get("center_y", 0.0),
        center_z=settings.get("center_z", 0.0),
        size_x=settings.get("size_x", 30.0),
        size_y=settings.get("size_y", 30.0),
        size_z=settings.get("size_z", 30.0),
        task=settings.get("task", "screen"),
        search_mode=settings.get("search_mode", "balance"),
        exhaustiveness=advanced.get("exhaustiveness", -1),
        randomize=advanced.get("randomize", True),
        mc_steps=advanced.get("mc_steps", -1),
        opt_steps=advanced.get("opt_steps", -1),
        refine_steps=advanced.get("refine_steps", 5),
        num_pose=advanced.get("num_pose", 10),
        rmsd_limit=advanced.get("rmsd_limit", 1.0),
        energy_range=advanced.get("energy_range", 5.0),
        seed=advanced.get("seed", 1234567),
        use_tor_lib=advanced.get("tor_lib", False),
        constraint_docking=settings.get("constraint_docking", False),
        gpu_device_id=hardware.get("gpu_device_id", 0),
        max_gpu_mem=hardware.get("max_gpu_memory", 0)

    )
    
    # Set receptor
    print("Setting receptor...")
    docking_pipeline.set_receptor(receptor_info)
    
    # Add ligands
    print("Adding ligands...")
    docking_pipeline.add_ligands(ligands_info)
    
    # Run docking
    print("\nRunning docking...")
    docking_pipeline.run()
    
    print("\n✓ Docking completed!")
    print(f"Results saved to: {output_dir}")


def test_pipeline():
    """Test the pipeline module"""
    print("\n" + "=" * 60)
    run_docking(example_dir)
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()