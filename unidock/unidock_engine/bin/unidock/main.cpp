#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include <vector> // ligand paths
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>

#include <unistd.h>
#include "myutils/errors.h"
#include "model/model.h"
#include "format/json.h"
#include "screening/screening.h"
#include "screening/core.h"

#include "main.h"

#include "constants/constants.h"


void print_help(){
    const char* STR_HELP = R"(
Usage: ud2 [options] [config_path]

Options:
  --version             Show version information
  --help                Show this help message
  --dump_config [path]  Dump default config to file (default: default.yaml)
  --log <path>          Specify log file path, only take effects when [config_path] is given
)";
    std::cout << STR_HELP;
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

void dump_config_template(const std::string& p){
    std::ofstream f(p);
    if (!f){
        std::cout << "Failed to create config template file: " << p.c_str() << "\n";
        exit(1);
    }

    f << STR_CONFIG_TEMPLATE;
}



int main(int argc, char* argv[])
{
    print_sign();
    print_version();
#ifdef DEBUG
    int log_level = 0;
#else
    int log_level = 1;
#endif
    std::string fp_log = "ud.log"; // default under working directory
    std::string fp_config;

    // parse arguments
    for (int i = 1; i < argc; ++i){
        std::string arg = argv[i];
        if (arg == "--version"){
            print_version();
            return 0;
        }
        if (arg == "--help"){
            print_help();
            return 0;
        }
        if (arg == "--dump_config"){
            std::string fp_dump = "default.yaml";
            if ((i + 1 < argc) && (argv[i + 1][0] != '-')){
                fp_dump = argv[++i];
            }
            dump_config_template(fp_dump);
            return 0;
        }

        if (arg == "--log"){
            if (argc > i + 1){
                fp_log = argv[i + 1];
                i += 1; // log parameter done
            } else{
                print_help();
                printf("--log requires a log path\n");
                return 1;
            }
        } else{
            if (fp_config.empty()){
                fp_config = arg;
                i ++;
            } else{
                print_help();
                printf("After config path is parsed as %s, unexpected argument: %s\n", fp_config.c_str(), arg.c_str());
                return 1;
            }
        }
    }

    if (fp_config.empty()){
        print_help();
        printf("Missing argument: config file path\n");
        exit(1);
    }

    init_logger(fp_log, log_level);
    spdlog::info("Using config file: {}", fp_config);


    spdlog::info("==================== UD2 Starts! ======================\n");
    auto start = std::chrono::high_resolution_clock::now();

    int mc_steps = 0;
    DockParam dock_param;
    std::string fp_score;

    YAML::Node config = YAML::LoadFile(fp_config);
    CoreContext ctx;


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
    bool use_tor_lib = get_config_with_err<bool>(config, "Advanced", "tor_lib", false);;
    if (use_tor_lib){
        spdlog::warn("Torsion Library is used.");
    }else{
        spdlog::warn("Torsion Library is NOT used.");
    }

    // todo: write into constants.h
    Box box_protein;
    box_protein.x_lo = dock_param.box.x_lo - VINA_CUTOFF;
    box_protein.x_hi = dock_param.box.x_hi + VINA_CUTOFF;
    box_protein.y_lo = dock_param.box.y_lo - VINA_CUTOFF;
    box_protein.y_hi = dock_param.box.y_hi + VINA_CUTOFF;
    box_protein.z_lo = dock_param.box.z_lo - VINA_CUTOFF;
    box_protein.z_hi = dock_param.box.z_hi + VINA_CUTOFF;

    read_ud_from_json(fp_json, box_protein, fix_mol, flex_mol_list, fns_flex, use_tor_lib);
    spdlog::info("Receptor has {:d} atoms in box", fix_mol.natom);
    spdlog::info("Flexible molecules count: {:d}", flex_mol_list.size());
    if (flex_mol_list.size() == 0){
        spdlog::error("No flexible molecules found in {}", fp_json);
    }

    // get total memory in MB and leave 5%
    float max_memory = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE) / 1024 / 1024 * 0.95;

    ctx.gpu_device_id = get_config_with_err<int>(config, "Hardware", "gpu_device_id", 0);


    // Advanced
    dock_param.num_pose = get_config_with_err<int>(config, "Advanced", "num_pose", dock_param.num_pose);
    dock_param.rmsd_limit = get_config_with_err<Real>(config, "Advanced", "rmsd_limit", dock_param.rmsd_limit);
    dock_param.energy_range = get_config_with_err<Real>(config, "Advanced", "energy_range", dock_param.energy_range);
    dock_param.seed = get_config_with_err<int>(config, "Advanced", "seed", dock_param.seed);


    // -------------------------------  Parse Settings -------------------------------
    std::string search_mode = get_config_with_err<std::string>(config, "Settings", "search_mode", "balance");
    if (search_mode == "fast"){
        dock_param.exhaustiveness = 128;
        dock_param.mc_steps = 20;
        dock_param.opt_steps = -1;
    } else if (search_mode == "balance"){
        dock_param.exhaustiveness = 256;
        dock_param.mc_steps = 30;
        dock_param.opt_steps = -1;
    } else if (search_mode == "detail"){
        dock_param.exhaustiveness = 512;
        dock_param.mc_steps = 40;
        dock_param.opt_steps = -1;
    } else if (search_mode == "free"){
        //
    } else{
        spdlog::critical("Not supported search_mode: {} doesn't belong to (fast, balance, detail, free)" , search_mode);
        exit(1);
    }

    dock_param.constraint_docking = get_config_with_err<bool>(config, "Settings", "constraint_docking", false);
    if (dock_param.constraint_docking){
        dock_param.randomize = false;
    }

    std::string bias = get_config_with_err<std::string>(config, "Advanced", "bias", "no");
    if (bias == "no"){
        dock_param.bias_type = BT_NO;
    }else if (bias == "pos"){
        dock_param.bias_type = BT_POS;
    }else if (bias == "align"){
        dock_param.bias_type = BT_ALIGN;
    }else{
        spdlog::critical("Not supported bias: {} doesn't belong to (no, pos, align)" , bias);
        exit(1);
    }

    dock_param.bias_k = get_config_with_err<Real>(config, "Advanced", "bias_k", dock_param.bias_k);


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

    ctx.task = task;
    ctx.dock_param = dock_param;
    ctx.fix_mol = std::move(fix_mol);
    ctx.flex_mol_list = std::move(flex_mol_list);
    ctx.fns_flex = std::move(fns_flex);
    ctx.output_dir = dp_out;
    ctx.max_memory = max_memory;
    ctx.name_json = name_json;
    core_pipeline(ctx);

    std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
    spdlog::info("UD2 Total Cost: {:.1f} ms", duration.count());

    return 0;
}
