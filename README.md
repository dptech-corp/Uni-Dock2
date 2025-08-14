# Uni-Dock2
GPU-accelerated molecular docking software: Uni-Dock 2

---
# Installation
## Conda Installation
The easiest way to install UniDock2 is via conda:

**Prerequisites**
* `Python = 3.10`
* `CUDA >= 12.0`

```sh
#You can modify the cuda-version to fit your environment.
conda install unidock2 cuda-version=12.0 -c http://quetz.dp.tech:8088/get/baymax -c conda-forge --no-repodata-use-zst 
```

## Manual Build
```sh
git clone https://github.com/dptech-corp/Uni-Dock2.git
```

### Step 1. Build unidock_engine
**Prerequisites**
* `CUDA toolkit >= 12.0 (Including nvcc)`
* `CMake >= 3.27`
* `C++ compiler`
* `Pybind11`

```sh
cd unidock/unidock_engine
pip install .
cd ../..
```

### Step 2. Build unidock_processing
```sh
conda install pyyaml pathos numpy pandas scipy networkx rdkit mdanalysis pdbfixer openmm cuda-version=12.0 msys_viparr_lpsolve55 ambertools_stable -c http://quetz.dp.tech:8088/get/baymax -c conda-forge --no-repodata-use-zst
pip install .
```

## Verify Installation
```sh
unidock2 --version
```

---
# Usage
Check `unidock2` usage by `unidock2 --help`. Basically there are two types of task: molecular docking and protein preparation. This document focuses on the docking task, since protein preparation is similar and we recommend consulting the help information.

## Configuration File
A configuration YAML file is all you need to run docking tasks:
```
unidock2 docking -cf your_config.yaml
```
Use `unidock2 docking --help` to check how to write the YAML file. 

**ATTENTION If a parameter is not written in the YAML, the default value of the parameter will be used (e.g., `size=[30, 30, 30]`). Carefully check the default values in the help information.**


## Command Line Parameters
Core parameters in the YAML file have command line equivalents. Command line inputs will override YAML values:
```sh
  -r RECEPTOR, --receptor RECEPTOR
                        Receptor structure file in PDB or DMS format
  -l LIGAND, --ligand LIGAND
                        Single ligand structure file in SDF format
  -lb LIGAND_BATCH, --ligand_batch LIGAND_BATCH
                        Recorded batch text file of ligand SDF file path
  -c center_x center_y center_z, --center center_x center_y center_z
                        Docking box center coordinates
  -o OUTPUT_DOCKING_POSE_SDF_FILE_NAME, --output_docking_pose_sdf_file_name OUTPUT_DOCKING_POSE_SDF_FILE_NAME
                        Output docking pose SDF file name
```

---
# Quick Docking Tutorial
A typical docking input includes at least one **receptor** file, one **ligand** file, docking pocket **center** coordinates and **box size**. Example cases could be found in the `examples` folder.

## 1. Free Docking
The ligand molecule can translate, rotate and adjust torsion angles within the docking box.

### 1.1. Molecular Docking
Single receptor vs. single ligand.

```
cd examples/free_docking/molecular_docking
```

**YAML**
Write the `test.yaml` as
```yaml
Required:
  receptor: 5WIU_protein_water_cleaned.pdb
  ligand: actives_cleaned.sdf
  center: [5.122, 18.327, 37.332]
Settings:
  box_size: [30.0, 30.0, 30.0]
```
and run `unidock2 docking -cf test.yaml`.

**Command Line**
You can also use command line parameters:
```sh
unidock2 docking -r 1G9V_protein_water_cleaned.pdb -l ligand_prepared.sdf -c 5.122 18.327 37.332
```


### 1.2. Virtual Screening
Single receptor vs. multiple ligands.

### 1.2.1 Single SDF with Multiple Ligands
```sh
cd examples/free_docking/virtual_screening
```

**YAML**
Write the `test.yaml` as
```yaml
Required:
  receptor: 5WIU_protein_cleaned.pdb
  ligand: actives_cleaned.sdf # One SDF file contains multiple ligands
  center: [5.122, 18.327, 37.332]
Settings:
  box_size: [30.0, 30.0, 30.0]
```
and run `unidock2 docking -cf test.yaml`.


**Command Line**
You can also use command line parameters:
```sh
unidock2 -r 5WIU_protein_cleaned.pdb -l actives_cleaned.sdf -c -18.0 15.2 -17.0
```

### 1.2.2 Multiple SDF Files
Use an index file to record SDF file names, like `test.index`
```sh
1.sdf
2.sdf
3.sdf
4.sdf
```

**YAML**
Then write the `test.yaml` as
```yaml
Required:
  receptor: 5WIU_protein_cleaned.pdb
  ligand_batch: test.index 
  center: [5.122, 18.327, 37.332]
Settings:
  box_size: [30.0, 30.0, 30.0]
```
and run `unidock2 docking -cf test.yaml`.

**Command Line**
```sh
unidock2 -r 5WIU_protein_cleaned.pdb -lb test.index -c -18.0 15.2 -17.0
```

### 1.2.3 Combined Input
SDF files from both `ligand` and `ligand_batch` sources will be processed.


## 2. Template Docking
When using a reference molecule, the query ligand will align to it. You need to set `template_docking = true`.

After alignment and during docking, the query can't translate or rotate. Only non-core torsions can be adjusted.

#### 2.1 Automatic Atom Mapping
Uni-Dock2 will automatically compute the atom mapping.
```
cd examples/constraint_docking/automatic_atom_mapping
```

**YAML**
Then write the `test.yaml` as
```yaml
Required:
  receptor: Bace.pdb
  ligand_batch: batch.dat
  center: [14.786, -0.626, -1.088]
Settings:
  box_size: [30.0, 30.0, 30.0]
Preprocessing:
  template_docking: true
  reference_sdf_file_name: reference.sdf
```
and run `unidock2 docking -cf test.yaml`


#### 2.2 Custom Atom Mapping
Specify `core_atom_mapping_dict_list` in the YAML file. 

**ATTENTION If length of `core_atom_mapping_dict_list` is smaller than ligand count, the remaining ligands will use automatically computed atom mapping instead.**

```
cd examples/constraint_docking/manual_atom_mapping
```

**YAML**
Then write the `test.yaml` as
```yaml
Required:
  receptor: protein.pdb
  ligand: ligand.sdf
  center: [9.028, 0.804, 21.789]
Settings:
  box_size: [30.0, 30.0, 30.0]
Preprocessing:
  template_docking: true
  reference_sdf_file_name: reference.sdf
  core_atom_mapping_dict_list: [{'0': 14,
    '1': 15,
    '10': 11,
    '11': 12,
    '12': 13,
    '16': 1,
    '17': 2,
    '18': 3,
    '19': 4,
    '20': 6,
    '21': 7,
    '22': 8,
    '23': 9,
    '24': 20,
    '25': 21,
    '27': 27,
    '6': 24,
    '7': 0,
    '8': 5,
    '9': 10}]
```
and run `unidock2 docking -cf test.yaml`


## 3. Covalent Docking
For covalent docking, please set `covalent_ligand = true` and specify `covalent_residue_atom_info_list`. `covalent_residue_atom_info_list` is a list of 3 tuple, specifying protein residue information of warhead, covalent bond starting atom, and covalent ending atom.

**ATTENTION** Input files **MUST** be prepared using [Hermite ligand preparation](https://hermite.dp.tech/login).


```
cd examples/covalent_docking/
```

**YAML**
Then write the `test.yaml` as
```yaml
Required:
  receptor: 1EWL_prepared.pdb
  ligand: covalent_mol.sdf
  center: [8.411, 13.047, 6.811]
Settings:
  box_size: [30.0, 30.0, 30.0]
Preprocessing:
  covalent_ligand: true
  covalent_residue_atom_info_list: [["", "CYX", 25, "CA"], ["", "CYX", 25, "CB"], ["", "CYX", 25, "SG"]]
```
and run `unidock2 docking -cf test.yaml`
