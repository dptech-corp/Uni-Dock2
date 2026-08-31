#ifndef UD2_PIPELINE_CORE_H
#define UD2_PIPELINE_CORE_H

#include <string>
#include <vector>
#include "format/json.h"
#include "model/model.h"

// The runtime adapter must populate every scalar field before calling core_pipeline().
// Initializers below are neutral safety values, not user-facing product defaults.
struct CoreInput {
    std::string bias;
    Real bias_k = 0;
    bool constraint_docking = false;
    int exhaustiveness = 0;
    int gpu_device_id = 0;
    Real energy_range = 0;
    int max_gpu_memory = 0;
    int mc_steps = 0;
    std::string name_json;
    int num_pose = 0;
    int opt_steps = 0;
    std::string output_dir;
    int refine_steps = 0;
    bool randomize = false;
    Real rmsd_limit = 0;
    int seed = 0;
    std::string search_mode;
    std::string task;
    bool energy_decomp = false;

    Box box;
    UDFixMol fix_mol;
    UDFlexMolList flex_mol_list;
    std::vector<std::string> fns_flex;
    PoseMap poses;
};


struct CoreContext {
    int max_memory = 0;
    std::string task;
    std::string output_dir;
    std::string name_json;
    bool energy_decomp = false;
    UDFixMol fix_mol;
    UDFlexMolList flex_mol_list;
    std::vector<std::string> fns_flex;
    DockParam dock_param;
};

int core_pipeline(CoreInput& ctx);

#endif // UD2_PIPELINE_CORE_H
