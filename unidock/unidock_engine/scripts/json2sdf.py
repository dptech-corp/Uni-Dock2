"""
This script is used to convert the json output of UD2 engine to sdf file.
The json file to tackle is the output of the UD2 engine.
The reference sdf file is the input to generate the input json of UD2 engine.

Usage:
python json2sdf.py <json_file> <reference_sdf> <output_sdf> [noH] [new_name]

Args:
    json_file: the json file to tackle
    reference_sdf: the reference sdf file
    output_sdf: the output sdf file
    noH: whether to remove hydrogen atoms
    new_name: the new name of the ligand

"""

import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign


def tran_json_to_sdf(fp_res_json, fp_sdf, fp_res_sdf, noH=False, new_name=None):
    # fixme: The atom order in fp_res_json is different from the original fp_sdf
    """ Use RDKit to perform an align&rmsd process, without Hydrogen """
    with open(fp_res_json, "r") as f:
        json_res = json.load(f)

    # fp_sdf is the input ligand structure for UD2 engine. By now, contains no Hydrogen atoms
    mol_ref = Chem.SDMolSupplier(str(fp_sdf), removeHs=noH)[0]
    for key in mol_ref.GetPropNames():
        mol_ref.ClearProp(key)

    ligand_name = list(json_res.keys())[0]
    assert len(ligand_name) > 0, "ligand_name not found!"
    if new_name is None:
        new_name = ligand_name

    # create a writer
    writer = Chem.SDWriter(fp_res_sdf)

    for idx, pose in enumerate(json_res[ligand_name]):
        mol_res = Chem.Mol(mol_ref)  # new copy for each pose
        mol_res.SetProp("_Name", f"{new_name}_pose_{idx}")
        conf = mol_res.GetConformer()

        coords_res = np.array(pose["coords"]).reshape(-1, 3)
        for i in range(conf.GetNumAtoms()):
            conf.SetAtomPosition(i, coords_res[i])

        # add energy property (assume energy value in pose dictionary)
        if "energy" in pose:
            # convert energy list to space-separated string
            energy_str = " ".join([f"{e:.3f}" for e in pose["energy"]])
            mol_res.SetProp("energy", energy_str)

        # output conf to sdf file
        writer.write(mol_res)

        # rmsd = rdMolAlign.CalcRMS(mol_ref, mol_res)
        # list_rmsd.append(rmsd)
    writer.close()
    print(f"The sdf file {fp_res_sdf} has been generated.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str, help="the json file to tackle")
    parser.add_argument("reference_sdf", type=str, help="the reference sdf file")
    parser.add_argument("output_sdf", type=str, help="the output sdf file")
    parser.add_argument("--noH", action="store_true", help="whether to remove hydrogen atoms")
    parser.add_argument("--new_name", type=str, help="the new name of the ligand")
    args = parser.parse_args()

    tran_json_to_sdf(args.json_file, args.reference_sdf, args.output_sdf, args.noH, args.new_name)
