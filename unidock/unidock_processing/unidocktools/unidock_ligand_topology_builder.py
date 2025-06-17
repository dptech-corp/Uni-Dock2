import os
import re
import dill as pickle
import logging
from multiprocess.pool import Pool

from rdkit import Chem
from rdkit.Chem import Descriptors, GetMolFrags, FragmentOnBonds
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from unidock_processing.ligand_topology.generic_rotatable_bond import (
    GenericRotatableBond,
)
from unidock_processing.ligand_topology import utils

from unidock_processing.torsion_library.torsion_library_driver import (
    TorsionLibraryDriver,
)
from unidock_processing.unidocktools.vina_atom_type import AtomType
from unidock_processing.unidocktools.unidock_vina_atom_types import VINA_ATOM_TYPE_DICT
from unidock_processing.unidocktools.unidock_ff_atom_types import FF_ATOM_TYPE_DICT


def build_molecular_graph(
    mol,
    torsion_library_dict,
    covalent_ligand=False,
    template_docking=False,
    reference_mol=None,
    core_atom_mapping_dict=None,
    working_dir_name=".",
):
    if covalent_ligand:
        mol, covalent_anchor_atom_info, _ = (
            utils.prepare_covalent_ligand_mol(mol)
        )
    else:
        covalent_anchor_atom_info = None

    if template_docking:
        if core_atom_mapping_dict is None:
            core_atom_mapping_dict = utils.get_template_docking_atom_mapping(
                reference_mol, mol
            )
        else:
            if not utils.check_manual_atom_mapping_connection(
                reference_mol, mol, core_atom_mapping_dict
            ):
                raise ValueError(
                    "Specified core atom mapping makes unconnected fragments!!"
                )

        core_atom_idx_list = utils.get_core_alignment_for_template_docking(
            reference_mol, mol, core_atom_mapping_dict
        )

        ## The tail parts of atoms in the core are cancelled and not beloings to the core itself. Currently this strategy is disabled temporarily. #noqa
        # for core_atom_idx in core_atom_idx_list:
        #     core_atom = mol.GetAtomWithIdx(core_atom_idx)
        #     for neighbor_atom in core_atom.GetNeighbors():
        #         if neighbor_atom.GetIdx() not in core_atom_idx_list:
        #             core_atom_idx_list.remove(core_atom_idx)
        #             break

        temp_template_aligned_ligand_sdf_file_name = os.path.join(
            working_dir_name, "ligand_template_aligned.sdf"
        )
        writer = Chem.SDWriter(temp_template_aligned_ligand_sdf_file_name)
        writer.write(mol)
        writer.close()

    else:
        core_atom_idx_list = None

    atom_typer = AtomType()
    atom_typer.assign_atom_types(mol)
    ComputeGasteigerCharges(mol)
    utils.assign_atom_properties(mol)

    rotatable_bond_finder = GenericRotatableBond()
    rotatable_bond_info_list = rotatable_bond_finder.identify_rotatable_bonds(mol)

    ## Construct nonbonded force atom pairs
    ##############################################################################
    atom_pair_12_13_nested_list, atom_pair_14_nested_list = (
        utils.calculate_nonbonded_atom_pairs(mol)
    )

    ## Construct gaff2 atom type and parameters
    ##############################################################################
    temp_ligand_sdf_file_name = os.path.join(working_dir_name, "ligand.sdf")
    writer = Chem.SDWriter(temp_ligand_sdf_file_name)
    writer.write(mol)
    writer.close()

    (
        atom_type_list,
        partial_charge_list,
        atom_parameter_dict,
        torsion_parameter_nested_dict,
    ) = utils.record_gaff2_atom_types_and_parameters(
        temp_ligand_sdf_file_name, "gas", working_dir_name
    )
    ###############################################################################

    ## Freeze bonds in core part to be unrotatable for template docking case
    ###############################################################################
    if template_docking:
        filtered_rotatable_bond_info_list = []
        for rotatable_bond_info in rotatable_bond_info_list:
            rotatable_begin_atom_idx = rotatable_bond_info[0]
            rotatable_end_atom_idx = rotatable_bond_info[1]
            if (
                rotatable_begin_atom_idx in core_atom_idx_list
                and rotatable_end_atom_idx in core_atom_idx_list
            ):
                continue
            else:
                filtered_rotatable_bond_info_list.append(rotatable_bond_info)

        rotatable_bond_info_list = filtered_rotatable_bond_info_list
    ###############################################################################

    bond_list = list(mol.GetBonds())
    rotatable_bond_idx_list = []
    for bond in bond_list:
        bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
        if (
            bond_info in rotatable_bond_info_list
            or bond_info_reversed in rotatable_bond_info_list
        ):
            rotatable_bond_idx_list.append(bond.GetIdx())

    if len(rotatable_bond_idx_list) != 0:
        splitted_mol = FragmentOnBonds(mol, rotatable_bond_idx_list, addDummies=False)
        splitted_mol_list = list(
            GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False)
        )
    else:
        splitted_mol_list = [mol]

    num_fragments = len(splitted_mol_list)

    ## Find fragment as the root node
    ##############################################################################
    root_fragment_idx = None
    if covalent_ligand:
        for fragment_idx in range(num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            for atom in fragment.GetAtoms():
                atom_info = (
                    atom.GetProp("chain_idx"),
                    atom.GetProp("residue_name"),
                    atom.GetIntProp("residue_idx"),
                    atom.GetProp("atom_name"),
                )
                if atom_info == covalent_anchor_atom_info:
                    root_fragment_idx = fragment_idx
                    break

            if root_fragment_idx is not None:
                break

        if root_fragment_idx is None:
            raise ValueError("Bugs in root finding code for covalent docking!")

    elif template_docking:
        for fragment_idx in range(num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            for atom in fragment.GetAtoms():
                internal_atom_idx = (
                    int(re.split(r"(\d+)", atom.GetProp("atom_name"))[1]) - 1
                )
                if internal_atom_idx in core_atom_idx_list:
                    root_fragment_idx = fragment_idx
                    break

            if root_fragment_idx is not None:
                break

        if root_fragment_idx is None:
            raise ValueError("Bugs in root finding code for template docking!")

    else:
        root_fragment_idx = utils.root_finding_strategy(
            splitted_mol_list, rotatable_bond_info_list
        )
    ##############################################################################

    root_fragment = splitted_mol_list[root_fragment_idx]
    num_root_atoms = root_fragment.GetNumAtoms()

    root_atom_idx_list = []

    for root_atom_idx in range(num_root_atoms):
        root_atom = root_fragment.GetAtomWithIdx(root_atom_idx)
        root_atom_idx_list.append(root_atom.GetIntProp("internal_atom_idx"))

    ##############################################################################

    ##############################################################################
    ## Record fragment atom idx
    num_fragments = len(splitted_mol_list)
    fragment_atom_idx_nested_list = [None] * num_fragments
    for fragment_idx in range(num_fragments):
        fragment_mol = splitted_mol_list[fragment_idx]
        num_fragment_atoms = fragment_mol.GetNumAtoms()
        fragment_atom_idx_list = [None] * num_fragment_atoms

        for atom_idx in range(num_fragment_atoms):
            atom = fragment_mol.GetAtomWithIdx(atom_idx)
            fragment_atom_idx_list[atom_idx] = atom.GetIntProp("internal_atom_idx")

        fragment_atom_idx_nested_list[fragment_idx] = fragment_atom_idx_list

    ##############################################################################
    ##############################################################################
    ## Assign atom info
    num_total_atoms = mol.GetNumAtoms()
    atom_info_nested_list = [None] * num_total_atoms
    for atom_idx in range(num_total_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        ff_atom_type = atom_type_list[atom_idx]
        vina_atom_type = atom.GetProp("vina_atom_type")

        atom_info_list = [None] * 8
        atom_info_list[0] = atom.GetDoubleProp("x")
        atom_info_list[1] = atom.GetDoubleProp("y")
        atom_info_list[2] = atom.GetDoubleProp("z")
        atom_info_list[3] = VINA_ATOM_TYPE_DICT[vina_atom_type]
        atom_info_list[4] = FF_ATOM_TYPE_DICT[ff_atom_type]
        atom_info_list[5] = partial_charge_list[atom_idx]
        atom_info_list[6] = atom_pair_12_13_nested_list[atom_idx]
        atom_info_list[7] = atom_pair_14_nested_list[atom_idx]

        atom_info_nested_list[atom_idx] = atom_info_list

    ##############################################################################
    ##############################################################################
    ## Assign torsion FF parameters
    torsion_library_driver = TorsionLibraryDriver(
        mol, rotatable_bond_info_list, torsion_library_dict
    )
    torsion_library_driver.perform_torsion_matches()
    torsion_library_driver.identify_torsion_mobile_atoms(root_atom_idx_list[0])

    torsion_info_nested_list = [None] * torsion_library_driver.num_rotatable_bonds
    for torsion_idx in range(torsion_library_driver.num_rotatable_bonds):
        torsion_info_list = [None] * 5

        torsion_atom_idx_list = torsion_library_driver.torsion_atom_idx_nested_list[
            torsion_idx
        ]
        torsion_value_list = torsion_library_driver.original_torsion_value_list[
            torsion_idx
        ]
        torsion_range_list = (
            torsion_library_driver.enumerated_torsion_range_nested_list[torsion_idx]
        )
        torsion_mobile_atom_idx_list = (
            torsion_library_driver.mobile_atom_idx_nested_list[torsion_idx]
        )

        torsion_atom_type_i = atom_type_list[torsion_atom_idx_list[0]]
        torsion_atom_type_j = atom_type_list[torsion_atom_idx_list[1]]
        torsion_atom_type_k = atom_type_list[torsion_atom_idx_list[2]]
        torsion_atom_type_l = atom_type_list[torsion_atom_idx_list[3]]

        torsion_type = (
            torsion_atom_type_i,
            torsion_atom_type_j,
            torsion_atom_type_k,
            torsion_atom_type_l,
        )
        if torsion_type in torsion_parameter_nested_dict:
            corrected_torsion_type = torsion_type
        else:
            corrected_torsion_type = tuple(reversed(torsion_type))

        torsion_parameter_dict_list = torsion_parameter_nested_dict[
            corrected_torsion_type
        ]
        num_torsion_parameters = len(torsion_parameter_dict_list)
        torsion_parameter_nested_list = [None] * num_torsion_parameters

        for torsion_parameter_idx in range(num_torsion_parameters):
            torsion_parameter_dict = torsion_parameter_dict_list[torsion_parameter_idx]
            torsion_parameter_list = [None] * 4
            torsion_parameter_list[0] = torsion_parameter_dict["barrier_factor"]
            torsion_parameter_list[1] = torsion_parameter_dict["barrier_height"]
            torsion_parameter_list[2] = torsion_parameter_dict["periodicity"]
            torsion_parameter_list[3] = torsion_parameter_dict["phase"]
            torsion_parameter_nested_list[torsion_parameter_idx] = (
                torsion_parameter_list
            )

        torsion_info_list[0] = torsion_atom_idx_list
        torsion_info_list[1] = torsion_value_list
        torsion_info_list[2] = torsion_range_list
        torsion_info_list[3] = torsion_mobile_atom_idx_list
        torsion_info_list[4] = torsion_parameter_nested_list

        torsion_info_nested_list[torsion_idx] = torsion_info_list
    ##############################################################################
    ##############################################################################

    return (
        atom_info_nested_list,
        torsion_info_nested_list,
        root_atom_idx_list,
        fragment_atom_idx_nested_list,
    )


def batch_topology_builder_process(
    ligand_sdf_file_name,
    covalent_ligand,
    template_docking,
    reference_sdf_file_name,
    core_atom_mapping_dict_list,
    working_dir_name,
):
    torsion_library_pkl_file_name = os.path.join(
        os.path.dirname(__file__),
        "..",
        "torsion_library",
        "data",
        "torsion_library.pkl",
    )
    with open(torsion_library_pkl_file_name, "rb") as torsion_library_pkl_file:
        torsion_library_dict = pickle.load(torsion_library_pkl_file)

    batch_ligand_mol_list = list(
        Chem.SDMolSupplier(ligand_sdf_file_name, removeHs=False)
    )

    if template_docking:
        reference_mol = Chem.SDMolSupplier(reference_sdf_file_name, removeHs=False)[0]
    else:
        reference_mol = None

    num_batch_ligands = len(batch_ligand_mol_list)
    ligand_info_dict_list = [None] * num_batch_ligands

    for ligand_idx in range(num_batch_ligands):
        ligand_mol = batch_ligand_mol_list[ligand_idx]
        ligand_name = ligand_mol.GetProp("_Name")
        core_atom_mapping_dict = core_atom_mapping_dict_list[ligand_idx]
        (
            atom_info_nested_list,
            torsion_info_nested_list,
            root_atom_idx_list,
            fragment_atom_idx_nested_list,
        ) = build_molecular_graph(
            ligand_mol,
            torsion_library_dict,
            covalent_ligand,
            template_docking,
            reference_mol,
            core_atom_mapping_dict,
            working_dir_name,
        )

        ligand_info_dict = {}
        ligand_info_dict["ligand_name"] = ligand_name
        ligand_info_dict["atom_info"] = atom_info_nested_list
        ligand_info_dict["torsion_info"] = torsion_info_nested_list
        ligand_info_dict["root_atom_idx"] = root_atom_idx_list
        ligand_info_dict["fragment_atom_idx"] = fragment_atom_idx_nested_list
        ligand_info_dict_list[ligand_idx] = ligand_info_dict

#        if remove_temp_files:
#            temp_file_name = os.path.join(working_dir_name, "*")
#            os.system(f"rm -rf {temp_file_name}")

#    if remove_temp_files:
#        os.system(f"rm -rf {working_dir_name}")

    return ligand_info_dict_list


class UnidockLigandTopologyBuilder(object):
    def __init__(
        self,
        ligand_sdf_file_name_list,
        covalent_ligand=False,
        template_docking=False,
        reference_sdf_file_name=None,
        core_atom_mapping_dict_list=None,
        working_dir_name=".",
    ):
        self.ligand_sdf_file_name_list = ligand_sdf_file_name_list
        self.ligand_mol_list = []

        for ligand_sdf_file_name in self.ligand_sdf_file_name_list:
            ligand_mol_list = list(
                Chem.SDMolSupplier(ligand_sdf_file_name, removeHs=False)
            )
            for ligand_mol in ligand_mol_list:
                if ligand_mol is None:
                    logging.error("Incorrect bond orders for molecule!")
                    continue
                if Descriptors.NumRadicalElectrons(ligand_mol) > 0:
                    logging.error("Molecule contains atoms with radicals!")
                    continue

                self.ligand_mol_list.append(ligand_mol)

        self.num_ligands = len(self.ligand_mol_list)
        if self.num_ligands == 0:
            raise ValueError(
                "Zero valid molecule after checking when preprocessing ligands!!"
            )

        self.covalent_ligand = covalent_ligand
        self.template_docking = template_docking

        if reference_sdf_file_name is not None:
            self.reference_sdf_file_name = os.path.abspath(reference_sdf_file_name)
        else:
            self.reference_sdf_file_name = None

        self.core_atom_mapping_dict_list = core_atom_mapping_dict_list
        if self.core_atom_mapping_dict_list is None:
            self.core_atom_mapping_dict_list = [None] * self.num_ligands
        if len(self.core_atom_mapping_dict_list) != self.num_ligands:
            raise ValueError(
                "Number of user specified core atom mapping dicts does not match \
                    the number of input molecules!!"
            )

        if self.template_docking:
            if self.reference_sdf_file_name is None:
                raise ValueError(
                    "template docking mode specified without reference SDF file!!"
                )
        else:
            self.reference_sdf_file_name = None

        self.static_root = False
        if self.template_docking or self.covalent_ligand:
            self.static_root = True

        self.root_working_dir_name = os.path.abspath(working_dir_name)
        self.ligand_json_file_name = os.path.join(
            self.root_working_dir_name, "ligands_unidock2.json"
        )

        self.n_cpu = os.cpu_count()

    def generate_batch_ligand_topology(self):
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        raw_num_batches = self.n_cpu
        num_ligands_per_batch = int(self.num_ligands / raw_num_batches) + 1

        num_batches = 0
        batch_ligand_idx_tuple_list = []
        for batch_idx in range(raw_num_batches):
            begin_ligand_idx = num_ligands_per_batch * batch_idx
            end_ligand_idx = num_ligands_per_batch * (batch_idx + 1)
            num_batches += 1

            if end_ligand_idx >= self.num_ligands:
                end_ligand_idx = self.num_ligands
                batch_ligand_idx_tuple = (begin_ligand_idx, end_ligand_idx)
                batch_ligand_idx_tuple_list.append(batch_ligand_idx_tuple)
                break
            else:
                batch_ligand_idx_tuple = (begin_ligand_idx, end_ligand_idx)
                batch_ligand_idx_tuple_list.append(batch_ligand_idx_tuple)

        batch_ligand_topology_builder_pool = Pool(processes=num_batches)
        batch_ligand_topology_builder_results_list = [None] * num_batches

        for batch_idx in range(num_batches):
            batch_ligand_idx_tuple = batch_ligand_idx_tuple_list[batch_idx]
            batch_ligand_mol_list = self.ligand_mol_list[
                batch_ligand_idx_tuple[0] : batch_ligand_idx_tuple[1]
            ]
            batch_core_atom_mapping_dict_list = self.core_atom_mapping_dict_list[
                batch_ligand_idx_tuple[0] : batch_ligand_idx_tuple[1]
            ]
            working_dir_name = os.path.join(
                self.root_working_dir_name, f"ligand_batch_{batch_idx}"
            )
            os.mkdir(working_dir_name)

            batch_ligand_sdf_file_name = os.path.join(
                working_dir_name, "ligand_batch.sdf"
            )
            batch_ligand_writer = Chem.SDWriter(batch_ligand_sdf_file_name)
            for ligand_mol in batch_ligand_mol_list:
                batch_ligand_writer.write(ligand_mol)
                batch_ligand_writer.flush()

            batch_ligand_writer.close()

            batch_ligand_topology_builder_results = (
                batch_ligand_topology_builder_pool.apply_async(
                    batch_topology_builder_process,
                    args=(
                        batch_ligand_sdf_file_name,
                        self.covalent_ligand,
                        self.template_docking,
                        self.reference_sdf_file_name,
                        batch_core_atom_mapping_dict_list,
                        working_dir_name,
                    ),
                )
            )

            batch_ligand_topology_builder_results_list[batch_idx] = (
                batch_ligand_topology_builder_results
            )

        batch_ligand_topology_builder_pool.close()
        batch_ligand_topology_builder_pool.join()

        self.total_ligand_info_dict_list = []
        for batch_idx in range(num_batches):
            batch_ligand_topology_builder_results = (
                batch_ligand_topology_builder_results_list[batch_idx]
            )
            ligand_info_dict_list = batch_ligand_topology_builder_results.get()
            for ligand_info_dict in ligand_info_dict_list:
                self.total_ligand_info_dict_list.append(ligand_info_dict)

        if len(self.total_ligand_info_dict_list) != self.num_ligands:
            raise ValueError(
                "Collected number of batch ligands does not equal to \
                    real total number of input ligands!!"
            )

    def get_summary_ligand_info_dict(self):
        self.total_ligand_info_summary_dict = {}

        for ligand_idx in range(self.num_ligands):
            ligand_info_dict = self.total_ligand_info_dict_list[ligand_idx]
            ligand_name = ligand_info_dict["ligand_name"]
            atom_info_nested_list = ligand_info_dict["atom_info"]
            torsion_info_nested_list = ligand_info_dict["torsion_info"]
            root_atom_idx_list = ligand_info_dict["root_atom_idx"]
            fragment_atom_idx_nested_list = ligand_info_dict["fragment_atom_idx"]

            self.total_ligand_info_summary_dict[ligand_name] = {
                "atoms": atom_info_nested_list,
                "torsions": torsion_info_nested_list,
                "root_atoms": root_atom_idx_list,
                "fragment_atom_idx": fragment_atom_idx_nested_list,
            }
