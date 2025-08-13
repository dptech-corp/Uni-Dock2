# Uni-Dock2
GPU-accelerated molecular docking software: Uni-Dock 2

---

## Installation
### Method 1: Conda Installation
The easiest way to install UniDock2 is via conda:

#### Prerequisites
* Python = 3.10
* CUDA >= 12.0

```sh
#You can modify the cuda-version to fit your environment.
conda install unidock2 cuda-version=12.0 -c http://quetz.dp.tech:8088/get/baymax -c conda-forge --no-repodata-use-zst 
```

### Method 2: Manual Build
```sh
git clone https://github.com/dptech-corp/Uni-Dock2.git
```

#### 1. Build unidock_engine
##### Prerequisites
* CUDA toolkit >= 12.0 (Including nvcc)
* CMake >= 3.27
* C++ compiler
* Pybind11

```sh
cd unidock/unidock_engine
pip install .
cd ../..
```

#### 2. Build unidock_processing
```sh
conda install pyyaml pathos numpy pandas scipy networkx rdkit mdanalysis pdbfixer openmm cuda-version=12.0 msys_viparr_lpsolve55 ambertools_stable -c http://quetz.dp.tech:8088/get/baymax -c conda-forge --no-repodata-use-zst

pip install .
```

---
## Usage
We recommend all users use our command line functionality.
After installation, there will be a command line tool `unidock2`.

A proper docking inputs should at least contains one receptor file, one or more ligand file, the docking pocket center positions, and the configuration file specifying all the other optional parameters.
Although all the parameters ni the configuration file has default values, it is still recommend to check and specify the parameters from the YAML file.

### `unidock2` command line parameters:
* --receptor (-r): Receptor structure file in PDB or DMS format (default: None)
* --ligand (-l): Single ligand structure file in SDF format (default: None)
* --ligand_batch (-lb): Recorded batch text file of ligand SDF file path (default: None)
* --center (-c): Docking box center coordinates (default: [0.0, 0.0, 0.0])
* --configurations (-cf): Uni-Dock2 configuration YAML file recording all other options (default: None)
* --version (-v): Show unidock2 program version
* --help (-h): Show help message and exit

** Note: --ligand or --ligand_batch should have at least one specified.

### Configuration YAML parameters:
** Advanced docking engine specified options:
* exhaustiveness: MC candidates count (roughly proportional to time). If given, value for search_mode will be overridden. (default: 512)
* randomize: Whether to randomize input pose before performing the global search. (default: true)
* mc_steps: If given, value for search_mode will be overridden. (default: 20)
* opt_steps: Optimization steps after the global search; -1 to use heuristic strategy. (default: -1)
* refine_steps: Refinement steps after clustering. (default: 5)
* num_pose: Number of the finally generated poses to output. (default: 10)
* rmsd_limit: Minimum RMSD between output poses. (default: 1.0)
* energy_range: Maximum energy difference between output poses and the best pose. (default: 5.0)
* seed: Explicit random seed. (default: 1234567)
* use_tor_lib: True to use torsion library. (default: false)

** Hardware options:
* n_cpu: Number of maximun CPU cores used in ligand preprocessing. (default: null)
* gpu_device_id: GPU device ID. (default: 0)

** Docking Settings options:
* box_size: Docking box size in the X, Y and Z dimension respectively in Angstrom (default: [30.0, 30.0, 30.0]).
* task: screen | score | mc (default: screen)
*** screen: The most common mode, perform randomize (if true) + MC (mc_steps) + optimization (opt_steps) + cluster (if true) + refinement (refine_steps)
*** score: Only provide scores for input ligands, no searching or optimization
*** mc: Only perform pure MC, namely opt_steps=0; no refinement, neither

* search_mode: fast | balance | detail | free. Using recommended settings of exhaustiveness and search steps. (default: balance)

** Docking Preprocessing options:
* template_docking: Specified to true to perform constraint docking mode. (default: false)
* reference_sdf_file_name: Reference molecule SDF file name in constraint docking. (default: null)
* core_atom_mapping_dict_list: Reference to ligand atom mapping dict, please see example case for more detail. (default: null)
* covalent_ligand: Specified to true to perform covalent docking mode. (default: false)
* covalent_residue_atom_info_list: Covalent warhead information for covalent receptor residues, please see example case for more detail. (default: null)
* preserve_receptor_hydrogen: Preserve hydrogen atoms in receptor preparation protocol. (default: false)
* temp_dir_name: Docking temporary files working directory. (default: /tmp)
                 If this is not specified, the default intermidiate files would be saved in /tmp and removed after finishing docking protocol.

* output_docking_pose_sdf_file_name: Output docking ligand pose SDF file name for `docking` command mode. (default: unidock2_pose.sdf)
* output_receptor_dms_file_name: Output receptor DMS file name for `protein_prep` command mode. (default: receptor_parameterized.dms)


### Example case:
The default configuration YAML file is at `unidock/unidock_configurations.yaml`.
In `examples` folder, there are cases for free docking, constraint docking and covalent docking.

* Free docking

** molecular docking case:

Single ligand docking case. Please see `examples/free_docking/molecular_docking`, and run
```
unidock2 -r 1G9V_protein_water_cleaned.pdb -l ligand_prepared.sdf -c 5.122 18.327 37.332 -cf unidock_configurations.yaml
```


** virtual screening case:

Virtual screening docking case. Please see `examples/free_docking/virtual_screening`, and run
```
unidock2 -r 5WIU_protein_cleaned.pdb -l actives_cleaned.sdf -c -18.0 15.2 -17.0 -cf unidock_configurations.yaml
```


* Constraint docking

For constraint docking, users need to turn on `template_docking` and specify the file path of `reference_sdf_file_name` in configuration YAML file.

** automatic atom mapping case:

Use internal algorithms to calculate core atom mapping. Please see `examples/constraint_docking/automatic_atom_mapping`, and run
```
unidock2 -r Bace.pdb -lb ligand_batch.dat -c 14.786 -0.626 -1.088 -cf unidock_configurations.yaml
```


** manual atom mapping case:

Uni-Dock2 also supports customized atom mapping specification in case the default MCS algorithm does not perfroms well.
To do this, specify `core_atom_mapping_dict_list` in the configuration YAML file, please refer to `examples/constraint_docking/manual_atom_mapping/unidock_configurations.yaml` for example usage.
Please see `examples/constraint_docking/manual_atom_mapping`, and run
```
unidock2 -r protein.pdb -l ligand.sdf -c 9.028 0.804 21.789 -cf unidock_configurations.yaml
```


* Covalent docking

For covalent docking, users need to turn on `covalent_ligand` and specify the covalent residue information `covalent_residue_atom_info_list` in the configuration YAML file.
The `covalent_residue_atom_info_list` is a list consist of 3 tuple, specifying protein residue information of heavy atoms from warhead to covalent bond starting atom, and covalent ending atom.
Please see `examples/covalent_docking/unidock_configurations.yaml` for example usage.
Please see `examples/covalent_docking`, and run
```
unidock2 -r 1EWL_prepared.pdb -l covalent_mol.sdf -c 8.411 13.047 6.811 -cf unidock_configurations.yaml
```
