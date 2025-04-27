#include <iostream>
#include <filesystem>
#include <string>
#include <vector> // ligand paths
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>

#include <unistd.h>
#include "myutils/errors.h"
#include "model/model.h"
#include "format/json.h"
#include "screening/screening.h"



void printSign() {
    // ANSI Shadow
        std::cout << R"(
    ██╗ccc██╗██████╗c██████╗c
    ██║ccc██║██╔══██╗╚════██╗
    ██║ccc██║██║cc██║c█████╔╝
    ██║ccc██║██║cc██║██╔═══╝c
    ╚██████╔╝██████╔╝███████╗
    c╚═════╝c╚═════╝c╚══════╝
    )" << std::endl;

}

template<typename T>
T get_config_with_err(const YAML::Node& config, const std::string& section, const std::string& key,
             const T& default_value = T()) {
    try {
        if (!config[section] || !config[section][key]) {
            spdlog::warn("Config item {}.{} doesn't exist, using default value", section, key);
            return default_value;
        }
        return config[section][key].as<T>();
    } catch (const YAML::Exception& e) {
        spdlog::critical("Failed to read config item {}.{}: {}", section, key, e.what());
        exit(1);
    }
}


int main(int argc, char* argv[])
{
#ifdef DEBUG
    int log_level = 0;
#else
    int log_level = 1;
#endif
    init_logger("ud.log", log_level);
    std::cout << "UD2 Version " << VERSION_NUMBER << "\n";
    printSign();

    spdlog::info("==================== UD2 Starts! ======================\n");
    auto start = std::chrono::high_resolution_clock::now();

    int mc_steps = 0;
    DockParam dock_param;
    std::string fp_score;
    int ncpu = std::thread::hardware_concurrency();
    std::string config_file = "config.yaml"; // default config file path

    if (argc > 1) {
        config_file = argv[1];
    } else {
        spdlog::critical("Missing argument for config file path\n");
        exit(1);
    }
    spdlog::info("Using config file: {}", config_file);
    YAML::Node config = YAML::LoadFile(config_file);


    // -------------------------------  Parse Advanced -------------------------------
    dock_param.exhaustiveness = get_config_with_err<int>(config, "Advanced", "exhaustiveness", dock_param.exhaustiveness);;
    dock_param.randomize = get_config_with_err<bool>(config, "Advanced", "randomize", dock_param.randomize);
    dock_param.mc_steps = get_config_with_err<int>(config, "Advanced", "mc_steps", mc_steps);
    dock_param.opt_steps = get_config_with_err<int>(config, "Advanced", "opt_steps", dock_param.opt_steps);
    if (dock_param.opt_steps < 0){ //heuristic
        dock_param.opt_steps = -1;
        spdlog::info("Use heuristic method to decide opt_steps");
    }

    dock_param.refine_steps = get_config_with_err<int>(config, "Advanced", "refine_steps", dock_param.refine_steps);

    // box
    Real center_x = get_config_with_err<Real>(config, "Settings", "center_x");
    Real center_y = get_config_with_err<Real>(config, "Settings", "center_y");
    Real center_z = get_config_with_err<Real>(config, "Settings", "center_z");
    Real size_x = get_config_with_err<Real>(config, "Settings", "size_x");
    Real size_y = get_config_with_err<Real>(config, "Settings", "size_y");
    Real size_z = get_config_with_err<Real>(config, "Settings", "size_z");
    dock_param.box.x_lo = center_x - size_x / 2;
    dock_param.box.x_hi = center_x + size_x / 2;
    dock_param.box.y_lo = center_y - size_y / 2;
    dock_param.box.y_hi = center_y + size_y / 2;
    dock_param.box.z_lo = center_z - size_z / 2;
    dock_param.box.z_hi = center_z + size_z / 2;

    std::string task = get_config_with_err<std::string>(config, "Settings", "task", "screen");

    // Input
    std::string fp_json = get_config_with_err<std::string>(config, "Inputs", "json");
    UDFlexMolList flex_mol_list;
    UDFixMol fix_mol;
    std::vector<std::string>fns_flex;

    if (fp_json.empty()){
        spdlog::critical("Empty json file path");
        exit(1);
    }
    // get input json name
    std::string name_json = std::filesystem::path(fp_json).filename().string();
    if (name_json.size() >= 5 && name_json.substr(name_json.size() - 5) == ".json") {
        name_json = name_json.substr(0, name_json.size() - 5);
    }

    // todo: remove these
    bool use_tor_lib = get_config_with_err<bool>(config, "Advanced", "tor_lib", true);;
    if (not use_tor_lib){
        spdlog::warn("Torsion Library is NOT used.");
    }
    dock_param.tor_prec = get_config_with_err<Real>(config, "Advanced", "tor_prec", 0.3);;
    spdlog::info("tor_prec: {}", dock_param.tor_prec);
    dock_param.box_prec = get_config_with_err<Real>(config, "Advanced", "box_prec", 1.0);;
    spdlog::info("box_prec: {}", dock_param.box_prec);

    // todo: write into constants.h
    Real cutoff = 8.0;
    Box box_protein;
    box_protein.x_lo = dock_param.box.x_lo - cutoff;
    box_protein.x_hi = dock_param.box.x_hi + cutoff;
    box_protein.y_lo = dock_param.box.y_lo - cutoff;
    box_protein.y_hi = dock_param.box.y_hi + cutoff;
    box_protein.z_lo = dock_param.box.z_lo - cutoff;
    box_protein.z_hi = dock_param.box.z_hi + cutoff;

    read_ud_from_json(fp_json, box_protein, fix_mol, flex_mol_list, fns_flex, use_tor_lib);
    spdlog::info("Receptor has {:d} atoms in box", fix_mol.natom);
    spdlog::info("Flexible molecules count: {:d}", flex_mol_list.size());
    if (flex_mol_list.size() == 0){
        spdlog::error("No flexible molecules found in {}", fp_json);
    }

    // get total memory in MB and leave 5%
    float max_memory = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE) / 1024 / 1024 * 0.95;
    int deviceCount = 0;
    
    checkCUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount > 0) {
        int device_id = get_config_with_err<int>(config, "Hardware", "gpu_device_id", 0);
        checkCUDA(cudaSetDevice(device_id));
        spdlog::info("Set GPU device id to {:d}", device_id);
        size_t avail, total;
        cudaMemGetInfo(&avail, &total);
        spdlog::info("Available Memory = {:d} MB   Total Memory = {:d} MB",
            avail / 1024 / 1024, total / 1024 / 1024);
        int max_gpu_memory = avail / 1024 / 1024 * 0.95; // leave 5%
        if (max_gpu_memory > 0 && max_gpu_memory < max_memory) {
            max_memory = (float) max_gpu_memory;
        }

    }else{
        spdlog::critical("No CUDA device is found!");
        exit(1);
    }

    // Advanced
    dock_param.num_pose = get_config_with_err<int>(config, "Advanced", "num_pose", dock_param.num_pose);
    dock_param.rmsd_limit = get_config_with_err<Real>(config, "Advanced", "rmsd_limit", dock_param.rmsd_limit);
    dock_param.energy_range = get_config_with_err<Real>(config, "Advanced", "energy_range", dock_param.energy_range);
    dock_param.seed = get_config_with_err<int>(config, "Advanced", "seed", dock_param.seed);


    // -------------------------------  Parse Settings -------------------------------
    std::string search_mode = get_config_with_err<std::string>(config, "Settings", "search_mode", "balance");
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
    if (dock_param.exhaustiveness < ncpu) {
        spdlog::warn("Low exhaustiveness doesn't utilize all CPUs");
    }

    dock_param.constraint_docking = get_config_with_err<bool>(config, "Settings", "constraint_docking", false);
    if (dock_param.constraint_docking){
        dock_param.randomize = false;
    }


    // -------------------------------  Perform Task -------------------------------
    std::string dp_out = get_config_with_err<std::string>(config, "Outputs", "dir");
    if (!std::filesystem::exists(dp_out)) {
        try {
            std::filesystem::create_directories(dp_out);
        } catch (const std::filesystem::filesystem_error& e) {
            spdlog::critical("Failed to create output directory {}: {}", dp_out, e.what());
            exit(1);
        }
    }

    if (task == "screen"){ // allow changing every parameter
        spdlog::info("----------------------- RUN Screening -----------------------");
        run_screening(fix_mol, flex_mol_list, fns_flex, dp_out, dock_param, max_memory, name_json);

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
        run_screening(fix_mol, flex_mol_list, fns_flex, dp_out, dock_param, max_memory, name_json);

    } else if (task == "benchmark_one"){
        spdlog::warn("benchmark task is not implemented");
        spdlog::info("----------------------- RUN Benchmark on One-Crystal-Ligand Cases -----------------------");
        spdlog::info("----------------------- Given poses are deemed as reference poses -----------------------");

    } else if (task == "mc"){
        dock_param.randomize = true;
        dock_param.opt_steps = 0;
        dock_param.refine_steps = 0;
        spdlog::info("----------------------- RUN Only Monte Carlo Random Walking -----------------------");
        run_screening(fix_mol, flex_mol_list, fns_flex, dp_out, dock_param, max_memory, name_json);

    } else{
        spdlog::critical("Not supported task: {} doesn't belong to (screen, local_only, mc)", task);
        exit(1);
    }

    std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
    spdlog::info("UD2 Total Cost: {:.1f} ms", duration.count());

    return 0;
}
