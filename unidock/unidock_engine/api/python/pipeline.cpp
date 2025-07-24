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
#include "format/pybind_parser.h"
#include "myutils/errors.h"
#include "screening/screening.h"


namespace py = pybind11;


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

        // Use PybindParser to parse receptor info
        PybindParser parser(receptor_info, py::dict());  // Empty dict for ligands, will be set separately
        parser.parse_receptor_info(box_protein, _fix_mol);
        spdlog::info("Receptor loaded: {:d} atoms in box", _fix_mol.natom);
    }

    void add_ligands(py::dict ligands_info) {
        // Use PybindParser to parse ligands info
        PybindParser parser(py::list(), ligands_info);  // Empty list for receptor, will be set separately
        parser.parse_ligands_info(_flex_mol_list, _fns_flex, _use_tor_lib);
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