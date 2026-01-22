#ifndef UD2_PIPELINE_CORE_H
#define UD2_PIPELINE_CORE_H

#include <string>
#include <vector>
#include "model/model.h"

// ============ Default values for docking parameters ============
namespace CoreInputDefaults {
    // Inherit defaults from DockParam
    inline const DockParam cid_dock_param{};
    inline const int seed = cid_dock_param.seed;
    inline const bool constraint_docking = cid_dock_param.constraint_docking;
    inline const int exhaustiveness = cid_dock_param.exhaustiveness;
    inline const bool randomize = cid_dock_param.randomize;
    inline const int mc_steps = cid_dock_param.mc_steps;
    inline const int opt_steps = cid_dock_param.opt_steps;
    inline const int refine_steps = cid_dock_param.refine_steps;
    inline const int num_pose = cid_dock_param.num_pose;
    inline const Real energy_range = cid_dock_param.energy_range;
    inline const Real rmsd_limit = cid_dock_param.rmsd_limit;
    inline const Real bias_k = cid_dock_param.bias_k;
    // CoreInput-specific defaults
    constexpr const char* bias = "no";
    constexpr const char* task = "screen";
    constexpr const char* search_mode = "balance";
    constexpr int gpu_device_id = 0;
    constexpr int max_gpu_memory = 0;
    constexpr bool use_tor_lib = false;
    constexpr const char* name_json = "from_python_obj";
}

struct CoreInput {
    int seed = CoreInputDefaults::seed;
    bool constraint_docking = CoreInputDefaults::constraint_docking;
    int exhaustiveness = CoreInputDefaults::exhaustiveness;
    bool randomize = CoreInputDefaults::randomize;
    int mc_steps = CoreInputDefaults::mc_steps;
    int opt_steps = CoreInputDefaults::opt_steps;
    int refine_steps = CoreInputDefaults::refine_steps;
    int num_pose = CoreInputDefaults::num_pose;
    Real energy_range = CoreInputDefaults::energy_range;
    Real rmsd_limit = CoreInputDefaults::rmsd_limit;
    std::string bias = CoreInputDefaults::bias;
    Real bias_k = CoreInputDefaults::bias_k;
    std::string task = CoreInputDefaults::task;
    std::string search_mode = CoreInputDefaults::search_mode;
    std::string output_dir;
    std::string name_json;
    int gpu_device_id = CoreInputDefaults::gpu_device_id;
    int max_gpu_memory = CoreInputDefaults::max_gpu_memory;
    Box box;
    UDFixMol fix_mol;
    UDFlexMolList flex_mol_list;
    std::vector<std::string> fns_flex;
};


struct CoreContext {
    std::string task;
    std::string output_dir;
    std::string name_json;
    float max_memory = 0.0f;
    UDFixMol fix_mol;
    UDFlexMolList flex_mol_list;
    std::vector<std::string> fns_flex;
    DockParam dock_param;
};

int core_pipeline(CoreInput& ctx);

void print_sign();

void print_version();


#endif // UD2_PIPELINE_CORE_H

