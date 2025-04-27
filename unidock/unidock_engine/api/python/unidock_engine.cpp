#include <string>
#include <vector>
#include <filesystem>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <spdlog/spdlog.h>

#include "model/model.h"
#include "format/json.h"
#include "myutils/errors.h"
#include "screening/screening.h"


namespace py = pybind11;

void RunDockingPipeline(
    // Input parameter (json file path)
    std::string json_file_path,

    // Output parameter
    std::string output_dir,

    // Settings parameters
    Real center_x, 
    Real center_y, 
    Real center_z,
    Real size_x, 
    Real size_y, 
    Real size_z,

    // task type
    std::string task = "screen",

    // search mode
    std::string search_mode = "balance",
    
    // dock parameters
    int exhaustiveness = 128,
    bool randomize = true,
    int mc_steps = 20,
    int opt_steps = 10,
    Real tor_prec = 0.3,
    Real box_prec = 1.0,
    int refine_steps = 5,
    int num_pose = 1,
    Real rmsd_limit = 1.0,
    Real energy_range = 3.0,
    int seed = 12345,

    // Advanced parameters
    bool constraint_docking = false,
    bool use_tor_lib = true,
    
    // Hardware parameters
    int gpu_device_id = 0
) {
    auto start = std::chrono::high_resolution_clock::now();

    // Check if JSON file exists
    if (!std::filesystem::exists(json_file_path)) {
        throw std::runtime_error("Input JSON file does not exist: " + json_file_path);
    }

    // Get input json name
    std::string name_json = std::filesystem::path(json_file_path).filename().string();
    if (name_json.size() >= 5 && name_json.substr(name_json.size() - 5) == ".json") {
        name_json = name_json.substr(0, name_json.size() - 5);
    }
    
    // Initialize docking parameters
    DockParam dock_param;
    
    // Set box parameters from center and size
    dock_param.box.x_lo = center_x - size_x / 2;
    dock_param.box.x_hi = center_x + size_x / 2;
    dock_param.box.y_lo = center_y - size_y / 2;
    dock_param.box.y_hi = center_y + size_y / 2;
    dock_param.box.z_lo = center_z - size_z / 2;
    dock_param.box.z_hi = center_z + size_z / 2;
    
    // Real cutoff for protein box
    Real cutoff = 8.0;
    Box box_protein;
    box_protein.x_lo = dock_param.box.x_lo - cutoff;
    box_protein.x_hi = dock_param.box.x_hi + cutoff;
    box_protein.y_lo = dock_param.box.y_lo - cutoff;
    box_protein.y_hi = dock_param.box.y_hi + cutoff;
    box_protein.z_lo = dock_param.box.z_lo - cutoff;
    box_protein.z_hi = dock_param.box.z_hi + cutoff;
    
    // Create data structures for molecules
    UDFixMol fix_mol;
    UDFlexMolList flex_mol_list;
    std::vector<std::string> fns_flex;
    
    // Read molecules from JSON
    read_ud_from_json(json_file_path, box_protein, fix_mol, flex_mol_list, fns_flex, use_tor_lib);
    spdlog::info("Receptor has {:d} atoms in box", fix_mol.natom);
    spdlog::info("Flexible molecules count: {:d}", flex_mol_list.size());
    if (flex_mol_list.size() == 0) {
        spdlog::error("No flexible molecules found in {}", json_file_path);
    }


    // get total memory in MB and leave 5%
    float max_memory = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE) / 1024 / 1024 * 0.95;
    int deviceCount = 0;
    checkCUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        spdlog::critical("No CUDA device is found!");
        exit(1);
    }
    checkCUDA(cudaSetDevice(gpu_device_id));
    spdlog::info("Set GPU device id to {:d}", gpu_device_id);
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    spdlog::info("Available Memory = {:d} MB   Total Memory = {:d} MB",
        avail / 1024 / 1024, total / 1024 / 1024);
    int max_gpu_memory = avail / 1024 / 1024 * 0.95; // leave 5%
    if (max_gpu_memory > 0 && max_gpu_memory < max_memory) {
        max_memory = (float) max_gpu_memory;
    }
    
    // Check and extract advanced parameters
    dock_param.exhaustiveness = exhaustiveness;
    dock_param.randomize = randomize;
    dock_param.mc_steps = mc_steps;
    dock_param.opt_steps = opt_steps;
    dock_param.tor_prec = tor_prec;
    dock_param.box_prec = box_prec;
    dock_param.refine_steps = refine_steps;
    dock_param.num_pose = num_pose;
    dock_param.rmsd_limit = rmsd_limit;
    dock_param.energy_range = energy_range;
    dock_param.seed = seed;

    if (search_mode == "fast"){
        dock_param.exhaustiveness = 64;
        dock_param.mc_steps = 30;
        dock_param.opt_steps = 3;
    } else if (search_mode == "balance"){
        dock_param.exhaustiveness = 64;
        dock_param.mc_steps = 200;
        dock_param.opt_steps = 5;
    } else if (search_mode == "detail"){
        dock_param.exhaustiveness = 512;
        dock_param.mc_steps = 300;
        dock_param.opt_steps = 5;
    } else if (search_mode == "free"){

    } else{
        spdlog::critical("Not supported search_mode: {} doesn't belong to (fast, balance, detail, free)" , search_mode);
        exit(1);
    }

    if (dock_param.exhaustiveness < std::thread::hardware_concurrency()) {
        spdlog::warn("Low exhaustiveness doesn't utilize all CPUs");
    }

    // Check whether constrain-docking
    dock_param.constraint_docking = constraint_docking;
    if (constraint_docking){
        dock_param.randomize = false;
    }
    
    // Create output directory if it doesn't exist
    if (!std::filesystem::exists(output_dir)) {
        try {
            std::filesystem::create_directories(output_dir);
        } catch (const std::filesystem::filesystem_error& e) {
            spdlog::critical("Failed to create output directory {}: {}", output_dir, e.what());
            exit(1);
        }
    }
    
    // Run the docking
    if (task == "screen"){ // allow changing every parameter
        spdlog::info("----------------------- RUN Screening -----------------------");
        run_screening(fix_mol, flex_mol_list, fns_flex, output_dir, dock_param, max_memory, name_json);

    } else if (task == "score"){
        spdlog::info("----------------------- RUN Only Scoring -----------------------");
        dock_param.randomize = false;
        dock_param.exhaustiveness = 1;
        dock_param.mc_steps = 0;
        dock_param.opt_steps = 0;
        dock_param.refine_steps = 0;
        dock_param.num_pose = 1;
        dock_param.energy_range = 999;
        dock_param.rmsd_limit = 999;
        run_screening(fix_mol, flex_mol_list, fns_flex, output_dir, dock_param, max_memory, name_json);

    } else if (task == "benchmark_one"){
        spdlog::warn("benchmark task is not implemented");
        spdlog::info("----------------------- RUN Benchmark on One-Crystal-Ligand Cases -----------------------");
        spdlog::info("----------------------- Given poses are deemed as reference poses -----------------------");

    } else if (task == "mc"){
        dock_param.randomize = true;
        dock_param.opt_steps = 0;
        dock_param.refine_steps = 0;
        spdlog::info("----------------------- RUN Only Monte Carlo Random Walking -----------------------");
        run_screening(fix_mol, flex_mol_list, fns_flex, output_dir, dock_param, max_memory, name_json);

    } else{
        spdlog::critical("Not supported task: {} doesn't belong to (screen, local_only, mc)", task);
        exit(1);
    }

    std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
    spdlog::info("UD2 Total Cost: {:.1f} ms", duration.count());
}

// Define Python module
PYBIND11_MODULE(unidock_engine, m) {
    m.doc() = "Python bindings for the Uni-Dock2 molecular docking engine";

    m.def(
        "run_docking_pipeline", 
        &RunDockingPipeline, 
        "Run docking pipeline using the Uni-Dock2 engine",
        py::arg("json_file_path"),
        py::arg("output_dir"),
        py::arg("center_x"),
        py::arg("center_y"),
        py::arg("center_z"),
        py::arg("size_x"),
        py::arg("size_y"),
        py::arg("size_z"),
        py::arg("task") = "screen",
        py::arg("search_mode") = "balance",
        py::arg("exhaustiveness") = 128,
        py::arg("randomize") = true,
        py::arg("mc_steps") = 20,
        py::arg("opt_steps") = 10,
        py::arg("tor_prec") = 0.3,
        py::arg("box_prec") = 1.0,
        py::arg("refine_steps") = 5,
        py::arg("num_pose") = 1,
        py::arg("rmsd_limit") = 1.0,
        py::arg("energy_range") = 3.0,
        py::arg("seed") = 12345,
        py::arg("constraint_docking") = false,
        py::arg("use_tor_lib") = true,
        py::arg("gpu_device_id") = 0
    );
}