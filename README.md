# Uni-Dock2
A GPU-accelerated molecular docking software incorporating substantial algorithmic enhancements to improve docking accuracy.

---
# Installation
## Conda Installation
The easiest way to install UniDock2 is via conda:

**Prerequisites**
* Linux x86_64 + NVIDIA GPU
* `Python = 3.10`
* `CUDA >= 12.0`

```sh
#You can modify the cuda-version to fit your environment.
conda install unidock2 cuda-version=12.0 -c http://quetz.dp.tech:8088/get/baymax -c conda-forge
```

## Manual Build
```sh
git clone https://github.com/dptech-corp/Uni-Dock2.git
```

### Build and install Uni-Dock2
**Prerequisites**
* `CUDA toolkit (Including nvcc)`
* `CMake >= 3.27`
* `C++ compiler`
* `Pybind11`

```sh
# BEGIN GENERATED: conda run dependencies
conda install numpy networkx pyyaml pydantic rdkit openmm pdbfixer msys_viparr_lpsolve55 ambertools_stable -c http://quetz.dp.tech:8088/get/baymax -c conda-forge --no-repodata-use-zst
# END GENERATED: conda run dependencies
pip install .
```

For an engine developer build that includes the private Python binding and tests:

```sh
cmake -S engine -B build/engine \
  -DBUILD_API=ON \
  -DBUILD_TEST=ON
cmake --build build/engine
ctest --test-dir build/engine --output-on-failure
```

## Verify Installation
```sh
unidock2 --version
```

---
# Usage
Check `unidock2` usage by `unidock2 --help`. The subcommands are `docking`, `prepare_protein`, and `prepare_ligands`. This document focuses on docking; the prepare commands reuse the same YAML/CLI resolution rules.

## Configuration File
A configuration YAML file is all you need to run docking tasks:
```
unidock2 docking -cf your_config.yaml
# or: unidock2 docking --config your_config.yaml
```
Use `unidock2 docking --help` to check how to write the YAML file. 

**ATTENTION If a parameter is not written in the YAML, the default value of the parameter will be used (e.g., `size=[30, 30, 30]`). Carefully check the default values in the help information.**


## Receptor and Ligand Inputs
* Receptor (`-r` / `Required.receptor`): PDB or DMS. A **DMS** file is treated as an already prepared receptor and **skips protein preparation**. Use `unidock2 prepare_protein` to turn a PDB into a reusable DMS.
* Ligand (`-l` / `Required.ligand`): a single SDF file, a directory of SDF files (non-recursive `*.sdf`), or a **UD2LIG** directory that contains `manifest.json` with magic `ud2lig`. UD2LIG skips ligand preprocessing. Produce one with `unidock2 prepare_ligands -l ... -o mylibrary.ud2lig`, or reuse the `{pose_stem}.ud2lig` directory that `unidock2 docking` writes next to `-o` / `--output_sdf` by default.
* Ligand batch (`-lb`) is unchanged: a text file with one SDF path per line. It can be combined with a single SDF or an SDF directory, but **not** with a UD2LIG directory.

## Working Directory
Intermediate files (receptor preparation chain, per-batch ligand topology, external tool logs) go to **one directory per run**, created under `unidock2_temp` beside the command output. There is nothing to configure: point `-o` at the disk you want and the intermediates follow. The absolute path is printed when the run starts.

* A **successful** run removes its working directory, and removes `unidock2_temp` too when nothing else is left in it.
* A **failed** run keeps the working directory so you can inspect the intermediates.
* `--keep_workdir` (`Preprocessing.keep_workdir`) keeps it even when the run succeeds.

```sh
# Intermediates go to /data/results/unidock2_temp/docking_<host>_<pid>_<timestamp>_<id>/
unidock2 docking -cf experiment.yaml -o /data/results/poses.sdf --keep_workdir
```

## Command Line Parameters
All supported parameters can be configured in YAML. Frequently changed scalar and short-list parameters also have command-line equivalents; explicit command-line inputs override YAML values. Run `unidock2 docking --help` for the generated list and current defaults.

Generate a complete YAML template with the current defaults and field comments:

```sh
unidock2 docking --dump_config
# Writes ./unidock2_config.yaml

# Or choose the output file explicitly:
unidock2 docking --dump_config my_config.yaml
```

For example, a reusable YAML configuration can be adjusted for one run without editing the file:
```sh
unidock2 docking -cf experiment.yaml \
  --seed 42 \
  --gpu_device_id 1 \
  --search_mode free \
  --exhaustiveness 1024
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

### 1.2.3 Directory of SDF Files
Point `-l` at a directory. Uni-Dock2 reads every `*.sdf` in that directory (not recursive). If the directory contains `manifest.json`, it is treated as a UD2LIG library instead.

### 1.2.4 Combined Input
SDF files from both `ligand` (file or SDF directory) and `ligand_batch` sources will be processed. A UD2LIG directory cannot be combined with `ligand_batch`.


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
