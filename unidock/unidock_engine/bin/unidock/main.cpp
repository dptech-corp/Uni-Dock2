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

    int mc_steps = 0;
    DockParam dock_param;
    std::string fp_score;

    YAML::Node config = YAML::LoadFile(fp_config);
    CoreContext ctx;


    // -------------------------------  Parse Advanced -------------------------------
    // Advanced
    ctx.num_pose = get_config_with_err<int>(config, "Advanced", "num_pose", dock_param.num_pose);
    ctx.rmsd_limit = get_config_with_err<Real>(config, "Advanced", "rmsd_limit", dock_param.rmsd_limit);
    ctx.energy_range = get_config_with_err<Real>(config, "Advanced", "energy_range", dock_param.energy_range);
    ctx.seed = get_config_with_err<int>(config, "Advanced", "seed", dock_param.seed);
    ctx.exhaustiveness = get_config_with_err<int>(config, "Advanced", "exhaustiveness", dock_param.exhaustiveness);;
    ctx.randomize = get_config_with_err<bool>(config, "Advanced", "randomize", dock_param.randomize);
    ctx.mc_steps = get_config_with_err<int>(config, "Advanced", "mc_steps", mc_steps);
    ctx.opt_steps = get_config_with_err<int>(config, "Advanced", "opt_steps", dock_param.opt_steps);
    ctx.refine_steps = get_config_with_err<int>(config, "Advanced", "refine_steps", dock_param.refine_steps);
    bool use_tor_lib = get_config_with_err<bool>(config, "Advanced", "tor_lib", false);;

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
    // dock_param.box.z_lo = center_z - size_z / 2;
    dock_param.box.z_hi = center_z + size_z / 2;

    ctx.task = get_config_with_err<std::string>(config, "Settings", "task", "screen");

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
        ctx.name_json = name_json.substr(0, name_json.size() - 5);
    }else{
        spdlog::error("Wrong JSON file name: ", name_json);
        exit(1);
    }


    read_ud_from_json(fp_json, dock_param.box, fix_mol, flex_mol_list, fns_flex, use_tor_lib);

    ctx.gpu_device_id = get_config_with_err<int>(config, "Hardware", "gpu_device_id", 0);



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
    ctx.output_dir = get_config_with_err<std::string>(config, "Outputs", "dir");
    ctx.dock_param = dock_param;
    ctx.fix_mol = std::move(fix_mol);
    ctx.flex_mol_list = std::move(flex_mol_list);
    ctx.fns_flex = std::move(fns_flex);
    core_pipeline(ctx);


    return 0;
}
