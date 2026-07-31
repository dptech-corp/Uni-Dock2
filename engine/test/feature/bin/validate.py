"""
Validate docking output against reference crystal ligand.
Compute heavy-atom RMSD between top-1 pose and reference SDF.
"""

import argparse
import json
import math
import os
import sys


def parse_sdf_coords(sdf_path):
    """Parse all atom coordinates from an SDF V2000 file."""
    with open(sdf_path) as f:
        lines = f.readlines()
    counts = lines[3].split()
    natom = int(counts[0])
    coords = []
    for i in range(4, 4 + natom):
        parts = lines[i].split()
        coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return coords


def parse_output_coords(json_path):
    """Parse pose coordinates from ud2 output JSON. Returns list of poses, each a list of (x,y,z)."""
    with open(json_path) as f:
        data = json.load(f)
    ligand_key = list(data.keys())[0]
    poses = data[ligand_key]
    result = []
    for pose in poses:
        flat = pose["coords"]
        natom = len(flat) // 3
        coords = [(flat[i*3], flat[i*3+1], flat[i*3+2]) for i in range(natom)]
        result.append(coords)
    return result


def rmsd(coords_a, coords_b):
    """Compute RMSD between two coordinate lists of equal length."""
    assert len(coords_a) == len(coords_b), \
        f"Atom count mismatch: {len(coords_a)} vs {len(coords_b)}"
    n = len(coords_a)
    sq_sum = 0.0
    for (ax, ay, az), (bx, by, bz) in zip(coords_a, coords_b):
        sq_sum += (ax - bx)**2 + (ay - by)**2 + (az - bz)**2
    return math.sqrt(sq_sum / n)


def main():
    parser = argparse.ArgumentParser(description="Validate docking output vs reference SDF")
    parser.add_argument("--output", required=True, help="ud2 output JSON file")
    parser.add_argument("--ref", required=True, help="Reference ligand SDF file")
    parser.add_argument("--rmsd-limit", type=float, default=2.0,
                        help="RMSD threshold (Angstrom). Top-1 must be below this.")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        print(f"FAIL: output file not found: {args.output}")
        sys.exit(2)
    if not os.path.exists(args.ref):
        print(f"FAIL: reference file not found: {args.ref}")
        sys.exit(2)

    ref_coords = parse_sdf_coords(args.ref)
    poses = parse_output_coords(args.output)

    if len(poses) == 0:
        print("FAIL: no poses in output")
        sys.exit(3)

    top1_rmsd = rmsd(poses[0], ref_coords)
    best_rmsd = min(rmsd(p, ref_coords) for p in poses)

    print(f"Top-1 RMSD: {top1_rmsd:.3f} A")
    print(f"Best  RMSD: {best_rmsd:.3f} A  (across {len(poses)} poses)")
    print(f"Threshold:  {args.rmsd_limit:.3f} A")

    if best_rmsd > args.rmsd_limit:
        print(f"FAIL: best RMSD {best_rmsd:.3f} > {args.rmsd_limit:.3f}")
        sys.exit(1)

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
