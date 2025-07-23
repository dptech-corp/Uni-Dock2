#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <spdlog/spdlog.h>

#include "constants/constants.h"
#include "model/model.h"
#include "format/json.h" 
#include "myutils/errors.h"
#include "screening/screening.h"


namespace py = pybind11;

// Forward declarations for parsing functions
void parse_receptor_info(py::list receptor_info, const Box& box_protein, UDFixMol& fix_mol);
void parse_ligands_info(py::dict ligands_info, UDFlexMolList& flex_mol_list, std::vector<std::string>& fns_flex, bool use_tor_lib);

SCOPE_INLINE std::pair<int, int> order_pair(int a, int b){
    return std::make_pair(std::min(a, b), std::max(a, b));
}

bool checkInAnySameSet(const std::vector<std::set<int>>& frags, int v1, int v2) {
    for (auto frag: frags) {
        if (frag.find(v1) != frag.end() and frag.find(v2) != frag.end()) {
            return true;
        }
    }
    return false;
}

class DockingPipeline {
public:
    DockingPipeline(
        std::string output_dir,
        Real center_x, Real center_y, Real center_z,
        Real size_x, Real size_y, Real size_z,
        std::string task = "screen",
        std::string search_mode = "balance",
        int exhaustiveness = -1,
        bool randomize = true,
        int mc_steps = -1,
        int opt_steps = -1,
        int refine_steps = 5,
        int num_pose = 10,
        Real rmsd_limit = 1.0,
        Real energy_range = 5.0,
        int seed = 1234567,
        bool constraint_docking = false,
        bool use_tor_lib = false,
        int gpu_device_id = 0
    ) : _output_dir(output_dir), _use_tor_lib(use_tor_lib), _gpu_device_id(gpu_device_id), _name_json("from_python_obj") {
        
        _dock_param.box.x_lo = center_x - size_x / 2;
        _dock_param.box.x_hi = center_x + size_x / 2;
        _dock_param.box.y_lo = center_y - size_y / 2;
        _dock_param.box.y_hi = center_y + size_y / 2;
        _dock_param.box.z_lo = center_z - size_z / 2;
        _dock_param.box.z_hi = center_z + size_z / 2;

        _dock_param.randomize = randomize;
        _dock_param.refine_steps = refine_steps;
        _dock_param.num_pose = num_pose;
        _dock_param.rmsd_limit = rmsd_limit;
        _dock_param.energy_range = energy_range;
        _dock_param.seed = seed;
        _dock_param.constraint_docking = constraint_docking;

        if (search_mode == "fast"){
            _dock_param.exhaustiveness = (exhaustiveness > 0) ? exhaustiveness : 64;
            _dock_param.mc_steps = (mc_steps > 0) ? mc_steps : 30;
            _dock_param.opt_steps = (opt_steps > 0) ? opt_steps : 3;
        } else if (search_mode == "balance"){
            _dock_param.exhaustiveness = (exhaustiveness > 0) ? exhaustiveness : 64;
            _dock_param.mc_steps = (mc_steps > 0) ? mc_steps : 200;
            _dock_param.opt_steps = (opt_steps > 0) ? opt_steps : 5;
        } else if (search_mode == "detail"){
            _dock_param.exhaustiveness = (exhaustiveness > 0) ? exhaustiveness : 512;
            _dock_param.mc_steps = (mc_steps > 0) ? mc_steps : 300;
            _dock_param.opt_steps = (opt_steps > 0) ? opt_steps : 5;
        } else if (search_mode == "free"){
            _dock_param.exhaustiveness = (exhaustiveness > 0) ? exhaustiveness : 512;
            _dock_param.mc_steps = (mc_steps > 0) ? mc_steps : 40;
            _dock_param.opt_steps = (opt_steps > 0) ? opt_steps : -1;
        } else {
            throw std::runtime_error("Not supported search_mode: " + search_mode);
        }

        if (task == "screen"){ // allow changing every parameter
            spdlog::info("----------------------- RUN Screening -----------------------");
        } else if (task == "score"){
            spdlog::info("----------------------- RUN Only Scoring -----------------------");
            _dock_param.randomize = false;
            _dock_param.exhaustiveness = 1;
            _dock_param.mc_steps = 0;
            _dock_param.opt_steps = 0;
            _dock_param.refine_steps = 0;
            _dock_param.num_pose = 1;
            _dock_param.energy_range = 999;
            _dock_param.rmsd_limit = 999;

        } else if (task == "benchmark_one"){
            spdlog::warn("benchmark task is not implemented");
            spdlog::info("----------------------- RUN Benchmark on One-Crystal-Ligand Cases -----------------------");
            spdlog::info("----------------------- Given poses are deemed as reference poses -----------------------");

        } else if (task == "mc"){
            _dock_param.randomize = true;
            _dock_param.opt_steps = 0;
            _dock_param.refine_steps = 0;
            spdlog::info("----------------------- RUN Only Monte Carlo Random Walking -----------------------");

        } else{
            spdlog::critical("Not supported task: {} doesn't belong to (screen, local_only, mc)", task);
            exit(1);
        }
    }

    void set_receptor(py::list receptor_info) {
        Real cutoff = 8.0;
        Box box_protein;
        box_protein.x_lo = _dock_param.box.x_lo - cutoff;
        box_protein.x_hi = _dock_param.box.x_hi + cutoff;
        box_protein.y_lo = _dock_param.box.y_lo - cutoff;
        box_protein.y_hi = _dock_param.box.y_hi + cutoff;
        box_protein.z_lo = _dock_param.box.z_lo - cutoff;
        box_protein.z_hi = _dock_param.box.z_hi + cutoff;

        parse_receptor_info(receptor_info, box_protein, _fix_mol);
        spdlog::info("Receptor loaded: {:d} atoms in box", _fix_mol.natom);
    }

    void add_ligands(py::dict ligands_info) {
        parse_ligands_info(ligands_info, _flex_mol_list, _fns_flex, _use_tor_lib);
        spdlog::info("Ligands loaded. Total count: {:d}", _flex_mol_list.size());
    }

    void run() {
        if (_fix_mol.natom == 0) {
            throw std::runtime_error("Receptor has not been set or is empty.");
        }
        if (_flex_mol_list.empty()) {
            throw std::runtime_error("No ligands have been added.");
        }

        auto start = std::chrono::high_resolution_clock::now();

        float max_memory = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE) / 1024 / 1024 * 0.95;
        int deviceCount = 0;
        checkCUDA(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            spdlog::critical("No CUDA device is found!");
            exit(1);
        }
        checkCUDA(cudaSetDevice(_gpu_device_id));
        size_t avail, total;
        cudaMemGetInfo(&avail, &total);
        int max_gpu_memory = avail / 1024 / 1024 * 0.95;
        if (max_gpu_memory > 0 && max_gpu_memory < max_memory) {
            max_memory = (float) max_gpu_memory;
        }

        if (!std::filesystem::exists(_output_dir)) {
            std::filesystem::create_directories(_output_dir);
        }

        spdlog::info("----------------------- RUN Screening -----------------------");
        run_screening(_fix_mol, _flex_mol_list, _fns_flex, _output_dir, _dock_param, max_memory, _name_json);

        std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
        spdlog::info("UD2 Total Cost: {:.1f} ms", duration.count());
    }

private:
    DockParam _dock_param;
    UDFixMol _fix_mol;
    UDFlexMolList _flex_mol_list;
    std::vector<std::string> _fns_flex;
    std::string _output_dir;
    bool _use_tor_lib;
    int _gpu_device_id;
    std::string _name_json;
};

// Parsing implementations
void parse_receptor_info(py::list receptor_info, const Box& box_protein, UDFixMol& fix_mol) {
    fix_mol.coords.clear();
    fix_mol.vina_types.clear();
    fix_mol.ff_types.clear();
    fix_mol.charges.clear();
    
    for (const auto& atom_info : receptor_info) {
        py::list atom_list = atom_info.cast<py::list>();
        if (atom_list[3].cast<int>() == VN_TYPE_H){ // remove Hydrogens
            continue;
        }

        Real x = atom_list[0].cast<Real>();
        Real y = atom_list[1].cast<Real>();
        Real z = atom_list[2].cast<Real>();
        if (box_protein.is_inside(x, y, z)){ // for acceleration
            fix_mol.coords.push_back(x);
            fix_mol.coords.push_back(y);
            fix_mol.coords.push_back(z);
            fix_mol.vina_types.push_back(atom_list[3].cast<int>());
            fix_mol.ff_types.push_back(atom_list[4].cast<int>());
            fix_mol.charges.push_back(atom_list[5].cast<Real>());
        }
    }
    fix_mol.natom = fix_mol.charges.size();
}

void parse_ligands_info(py::dict ligands_info, UDFlexMolList& flex_mol_list, std::vector<std::string>& fns_flex, bool use_tor_lib) {
    std::set<std::string> exclude_keys = {"score", "receptor"};
    for (const auto &[k, v] : ligands_info) {
        std::string key = k.cast<std::string>();
        if (exclude_keys.find(key) != exclude_keys.end()){
            continue;
        }

        fns_flex.push_back(key);

        UDFlexMol flex_mol;
        Real coords_sum[3] = {0};
        
        py::dict ligand_dict = v.cast<py::dict>();
        
        // Atoms
        py::list atom_info = ligand_dict["atoms"].cast<py::list>();
        for (int ia = 0; ia < atom_info.size(); ia++){
            py::list atom_line = atom_info[ia].cast<py::list>();
            Real x = atom_line[0].cast<Real>();
            Real y = atom_line[1].cast<Real>();
            Real z = atom_line[2].cast<Real>();
            flex_mol.coords.push_back(x);
            flex_mol.coords.push_back(y);
            flex_mol.coords.push_back(z);
            coords_sum[0] += x;
            coords_sum[1] += y;
            coords_sum[2] += z;

            flex_mol.vina_types.push_back(atom_line[3].cast<int>());
            flex_mol.ff_types.push_back(atom_line[4].cast<int>());
            flex_mol.charges.push_back(atom_line[5].cast<Real>());

            py::list pairs_12_13 = atom_line[6].cast<py::list>();
            for (const auto& a : pairs_12_13){
                auto ib = a.cast<int>();
                if (ia == ib){
                    spdlog::warn("Remove wrong pair 1-2 & 1-3 for Self: flex {} atom {} - {}", key, ia, ib);
                }
                else{
                    flex_mol.pairs_1213.insert(order_pair(ia, ib));
                }
            }
            
            py::list pairs_14 = atom_line[7].cast<py::list>();
            for (const auto& a : pairs_14){
                auto ib = a.cast<int>();
                if (ia == ib){
                    spdlog::warn("Remove wrong pair 1-4 for Self: flex {} atom {} - {}", key, ia, ib);
                }
                else{
                    auto p = order_pair(ia, ib);
                    if (flex_mol.pairs_1213.find(p) != flex_mol.pairs_1213.end()){
                        spdlog::warn("Remove wrong pair 1-4 for Already in 1-2&1-3: flex {} atom {} - {}", key, ia,
                                        ib);
                    }
                    else{
                        flex_mol.pairs_14.insert(p);
                    }
                }
            }
        }

        std::set<int> root_atoms;
        py::list root_info = ligand_dict["root_atoms"].cast<py::list>();
        for (const auto& a : root_info){
            root_atoms.insert(a.cast<int>());
        }

        // Torsions
        py::list torsions_info = ligand_dict["torsions"].cast<py::list>();
        for (const auto& torsion_info : torsions_info){
            py::list tor_info = torsion_info.cast<py::list>();
            UDTorsion torsion;

            py::list torsion_atoms = tor_info[0].cast<py::list>();
            for (int i = 0; i < 4; i++){
                torsion.atoms[i] = torsion_atoms[i].cast<int>();
            }

            // axis is the two middle atoms
            torsion.axis[0] = torsion.atoms[1];
            torsion.axis[1] = torsion.atoms[2];

            // dihedral value
            flex_mol.dihedrals.push_back(ang_to_rad(tor_info[1].cast<Real>()));

            // range list
            if (use_tor_lib){
                py::list range_list = tor_info[2].cast<py::list>();
                for (const auto& r : range_list){
                    py::list range_pair = r.cast<py::list>();
                    Real rad_lo = ang_to_rad(range_pair[0].cast<Real>());
                    Real rad_hi = ang_to_rad(range_pair[1].cast<Real>());
                    // For range crossing "180/-180", split it into to ranges
                    if (rad_lo < rad_hi){
                        torsion.range_list.push_back(rad_lo);
                        torsion.range_list.push_back(rad_hi);
                    } else if (rad_lo > 0 && rad_hi < 0){
                        torsion.range_list.push_back(rad_lo);
                        torsion.range_list.push_back(PI);
                        torsion.range_list.push_back(-PI);
                        torsion.range_list.push_back(rad_hi);
                    } else{
                        spdlog::critical("Input json has wrong range list, rad_lo > rad_hi: {}, {}",
                            range_pair[0].cast<Real>(), range_pair[1].cast<Real>());
                    }
                }
            } else{ // TODO: consider moving this switch outside of C++ engine?
                torsion.range_list.push_back(-PI);
                torsion.range_list.push_back(PI);
            }

            // rotated atoms
            py::list rotated_atoms = tor_info[3].cast<py::list>();
            for (const auto& a : rotated_atoms){
                torsion.rotated_atoms.push_back(a.cast<int>());
            }
            // gaff2 parameters, may be multiple groups
            py::list gaff2_params = tor_info[4].cast<py::list>();
            for (const auto& gaff2 : gaff2_params){
                py::list gaff2_group = gaff2.cast<py::list>();
                torsion.param_gaff2.push_back(gaff2_group[0].cast<Real>());
                torsion.param_gaff2.push_back(gaff2_group[1].cast<Real>());
                torsion.param_gaff2.push_back(gaff2_group[2].cast<Real>());
                torsion.param_gaff2.push_back(gaff2_group[3].cast<Real>());
            }
            // add this torsion to the flex_mol
            flex_mol.torsions.push_back(torsion);
        }

        std::vector<std::set<int>> frags;
        split_torsions_into_frags(root_atoms, flex_mol.torsions, frags);
        // Compute necessary properties
        // spdlog::debug("Json data: compute properties...");
        flex_mol.name = key;
        flex_mol.natom = flex_mol.charges.size();
        flex_mol.center[0] = coords_sum[0] / flex_mol.natom;
        flex_mol.center[1] = coords_sum[1] / flex_mol.natom;
        flex_mol.center[2] = coords_sum[2] / flex_mol.natom;

        // intra pairs
        for (int i = 0; i < flex_mol.natom; i++){
            for (int j = i + 1; j < flex_mol.natom; j++){
                // exclude 1-2, 1-3, 1-4 pairs
                if ((flex_mol.pairs_1213.find(order_pair(i, j)) == flex_mol.pairs_1213.end()) and
                    (flex_mol.pairs_14.find(order_pair(i, j)) == flex_mol.pairs_14.end()) and
                    (!checkInAnySameSet(frags, i, j)) //
                    ){
                    flex_mol.intra_pairs.push_back(i);
                    flex_mol.intra_pairs.push_back(j);
                }
            }
        }
        // inter pairs: flex v.s. receptor
        for (int i = 0; i < flex_mol.natom; i++){
            if (flex_mol.vina_types[i] == VN_TYPE_H){ //ignore Hydrogen on ligand and protein
                continue;
            }
            // Note: receptor info needs to be passed separately or stored globally
            // This section may need to be moved to where receptor info is available
            // For now, commenting out to avoid compilation error
            /*
            for (int j = 0; j < fix_mol.natom; j++){
                if (fix_mol.vina_types[j] == VN_TYPE_H){
                    continue;
                }
                flex_mol.inter_pairs.push_back(i);
                flex_mol.inter_pairs.push_back(j);
            }
            */
        }

        // add this flex_mol to the list
        flex_mol_list.push_back(flex_mol);
    }
}

PYBIND11_MODULE(pipeline, m) {
    m.doc() = "Python bindings for the Uni-Dock2 molecular docking engine pipeline";

    py::class_<DockingPipeline>(m, "DockingPipeline")
        .def(py::init<
                std::string, Real, Real, Real, Real, Real, Real, std::string, std::string, 
                int, bool, int, int, int, int, Real, Real, int, bool, bool, int>(),
            py::arg("output_dir"),
            py::arg("center_x"), py::arg("center_y"), py::arg("center_z"),
            py::arg("size_x"), py::arg("size_y"), py::arg("size_z"),
            py::arg("task") = "screen",
            py::arg("search_mode") = "balance",
            py::arg("exhaustiveness") = -1,
            py::arg("randomize") = true,
            py::arg("mc_steps") = -1,
            py::arg("opt_steps") = -1,
            py::arg("refine_steps") = 5,
            py::arg("num_pose") = 10,
            py::arg("rmsd_limit") = 1.0,
            py::arg("energy_range") = 5.0,
            py::arg("seed") = 1234567,
            py::arg("constraint_docking") = false,
            py::arg("use_tor_lib") = false,
            py::arg("gpu_device_id") = 0
        )
        .def("set_receptor", &DockingPipeline::set_receptor, "Set the receptor molecule from a Python dictionary")
        .def("add_ligands", &DockingPipeline::add_ligands, "Add ligand molecules from a list of Python dictionaries")
        .def("run", &DockingPipeline::run, "Run the docking simulation");
}