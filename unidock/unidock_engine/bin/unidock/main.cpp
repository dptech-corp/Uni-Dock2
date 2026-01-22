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

#ifdef DEBUG
    int log_level = 0;
#else
    int log_level = 1;
#endif
    std::string fp_log = "ud.log"; // default under working directory
    std::string fp_config;

    print_sign();

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
            printf("Dump config template file: %s\n", fp_dump.c_str());
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
        return 1;
    }

    print_version();

    init_logger(fp_log, log_level);
    spdlog::info("Using config file: {}", fp_config);

    YAML::Node config = YAML::LoadFile(fp_config);
    CoreInput cipt;

    // -------------------------------  Parse Advanced -------------------------------
    cipt.num_pose = get_config_with_err<int>(config, "Advanced", "num_pose", CoreInputDefaults::num_pose);
    cipt.rmsd_limit = get_config_with_err<Real>(config, "Advanced", "rmsd_limit", CoreInputDefaults::rmsd_limit);
    cipt.energy_range = get_config_with_err<Real>(config, "Advanced", "energy_range", CoreInputDefaults::energy_range);
    cipt.seed = get_config_with_err<int>(config, "Advanced", "seed", CoreInputDefaults::seed);
    cipt.exhaustiveness = get_config_with_err<int>(config, "Advanced", "exhaustiveness", CoreInputDefaults::exhaustiveness);
    cipt.randomize = get_config_with_err<bool>(config, "Advanced", "randomize", CoreInputDefaults::randomize);
    cipt.mc_steps = get_config_with_err<int>(config, "Advanced", "mc_steps", CoreInputDefaults::mc_steps);
    cipt.opt_steps = get_config_with_err<int>(config, "Advanced", "opt_steps", CoreInputDefaults::opt_steps);
    cipt.refine_steps = get_config_with_err<int>(config, "Advanced", "refine_steps", CoreInputDefaults::refine_steps);
    bool use_tor_lib = get_config_with_err<bool>(config, "Advanced", "tor_lib", CoreInputDefaults::use_tor_lib);
    cipt.bias = get_config_with_err<std::string>(config, "Advanced", "bias", std::string(CoreInputDefaults::bias));
    cipt.bias_k = get_config_with_err<Real>(config, "Advanced", "bias_k", CoreInputDefaults::bias_k);

    // -------------------------------  Parse Settings -------------------------------
    cipt.task = get_config_with_err<std::string>(config, "Settings", "task", std::string(CoreInputDefaults::task));
    cipt.search_mode = get_config_with_err<std::string>(config, "Settings", "search_mode", std::string(CoreInputDefaults::search_mode));
    cipt.constraint_docking = get_config_with_err<bool>(config, "Settings", "constraint_docking", CoreInputDefaults::constraint_docking);
    // box
    Real center_x = get_config_with_err<Real>(config, "Settings", "center_x");
    Real center_y = get_config_with_err<Real>(config, "Settings", "center_y");
    Real center_z = get_config_with_err<Real>(config, "Settings", "center_z");
    Real size_x = get_config_with_err<Real>(config, "Settings", "size_x");
    Real size_y = get_config_with_err<Real>(config, "Settings", "size_y");
    Real size_z = get_config_with_err<Real>(config, "Settings", "size_z");
    cipt.box.x_lo = center_x - size_x / 2;
    cipt.box.x_hi = center_x + size_x / 2;
    cipt.box.y_lo = center_y - size_y / 2;
    cipt.box.y_hi = center_y + size_y / 2;
    cipt.box.z_lo = center_z - size_z / 2;
    cipt.box.z_hi = center_z + size_z / 2;


    // -------------------------------  Parse Hardware -------------------------------
    cipt.gpu_device_id = get_config_with_err<int>(config, "Hardware", "gpu_device_id", CoreInputDefaults::gpu_device_id);
    cipt.max_gpu_memory = get_config_with_err<int>(config, "Hardware", "max_gpu_memory", CoreInputDefaults::max_gpu_memory);

    // -------------------------------  Parse Outputs -------------------------------
    cipt.output_dir = get_config_with_err<std::string>(config, "Outputs", "dir");

    // -------------------------------  Parse Inputs -------------------------------
    std::string fp_json = get_config_with_err<std::string>(config, "Inputs", "json");
    if (fp_json.empty()){
        spdlog::critical("Empty json file path");
        return 1;
    }
    // get input json name
    std::string name_json = std::filesystem::path(fp_json).filename().string();
    if (name_json.size() >= 5 && name_json.substr(name_json.size() - 5) == ".json") {
        cipt.name_json = name_json.substr(0, name_json.size() - 5);
    }else{
        spdlog::error("Wrong JSON file name: ", name_json);
        return 1;
    }

    UDFlexMolList flex_mol_list;
    UDFixMol fix_mol;
    std::vector<std::string>fns_flex;
    read_ud_from_json(fp_json, cipt.box, fix_mol, flex_mol_list, fns_flex, use_tor_lib);

    cipt.fix_mol = std::move(fix_mol);
    cipt.flex_mol_list = std::move(flex_mol_list);
    cipt.fns_flex = std::move(fns_flex);

    // -------------------------------  Run Pipeline -------------------------------
    if (core_pipeline(cipt) != 0) {
        spdlog::critical("Docking pipeline failed");
        return 1;
    }

    return 0;
}
