//
// Created by Congcong Liu on 24-9-23.
//

#ifndef DOCKTASK_H
#define DOCKTASK_H


#include "model/model.h"
#include <string>
#include <vector>
#include "score/vina.h"
#include "score/score.h"
#include "cuda/struct_array_manager.cuh"


/**
 * @brief Docking calculation model. Aiming to perform a docking computation method, thus
 * managing all related concepts, including the structure to compute on; the method to
 * apply, results to show...
 */
class DockTask{
public:
    // -------------------- Settings
    // Both - Parameters
    DockParam dock_param;
    int nflex = 0;
    bool show_score = true;
    bool energy_decomp = false;

    // -------------------- Molecules
    // CPU - Input Model
    UDFixMol udfix_mol;
    UDFlexMolList udflex_mols;
    std::vector<std::string> fns_flex;

    // CPU - Output Model
    std::string fp_json;

    // Construction

    DockTask(const UDFixMol& fix_mol, DockParam dock_param) :
        udfix_mol(fix_mol), dock_param(dock_param){
    };

    DockTask(const UDFixMol& fix_mol, const UDFlexMolList& flex_mol_list, DockParam dock_param,
             std::vector<std::string> fns_flex, std::string fp_json) :
        udfix_mol(fix_mol), udflex_mols(flex_mol_list), dock_param(dock_param), fns_flex(fns_flex), fp_json(fp_json){
        nflex = flex_mol_list.size();
    };

    void set_flex(const UDFlexMolList& flex_mol_list, DockParam dock_param,
                  std::vector<std::string> fns_flex,
                  std::string fp_json);

    /**
     * @brief Run a whole process: global search, cluster by RMSD, refinement by optimization and final output.
     */
    void run();

    // Create dpfix_mol and dpflex_mols
    void from_json(std::string fp); // todo: resume a task from file

    // Output
    void dump_poses_to_json(std::string fp_json);
    void free_fix_mol_gpu();

private:
    // CPU
    int n_atom_all_flex = 0;
    std::vector<std::vector<int>> clustered_pose_inds_list; 
    std::vector<std::vector<int>> filtered_pose_inds_list;
    // decomp_list[i_flex][i_pose] = per-atom inter energy decomposition
    std::vector<std::vector<std::vector<AtomEnergyDecomp>>> decomp_list;

    // GPU
    FixMol* fix_mol_cu;
    Real* fix_mol_real_cu;

    FlexPose* flex_pose_list_cu; // size: nflex * exhaustiveness
    StructArrayManager<FlexPose>* flex_pose_list_manager = nullptr;

    FlexTopo* flex_topo_list_cu; // size: nflex
    StructArrayManager<FlexTopo>* flex_topo_list_manager = nullptr;

    FlexPoseGradient* flex_grad_list_cu;
    FlexPoseHessian* flex_hessian_list_cu;

    FlexParamVina* flex_param_list_cu;
    StructArrayManager<FlexParamVina>* flex_param_list_manager = nullptr;
    
    FixParamVina* fix_param_cu;
    int* fix_param_int_cu;

    Real* aux_list_e_cu; // saves energy of all poses of all flexes, size: nflex * exhaustiveness
    int npose_clustered = 0;
    int* clustered_pose_inds_cu; // pose indices after clustering, -1 indicates that no pose is selected

    // GPU - auxiliary
    FlexPose* aux_poses_cu;
    StructArrayManager<FlexPose>* aux_poses_manager = nullptr;
    
    FlexPoseGradient* aux_grads_cu;
    StructArrayManager<FlexPoseGradient>* aux_grads_manager = nullptr;
    
    FlexPoseHessian* aux_hessians_cu;
    StructArrayManager<FlexPoseHessian>* aux_hessians_manager = nullptr;
    
    FlexForce* aux_forces_cu;
    StructArrayManager<FlexForce>* aux_forces_manager = nullptr;

    Real* aux_rmsd_matrix_cu; // E*E RMSD matrix on GPU, upper triangle filled by kernel


    // -------------------- Functions
    void prepare_vina();
    void run_search();
    void run_cluster();
    void run_refine();
    void run_filter();
    void run_score();
    void dump_poses();

    void alloc_gpu();
    void cp_to_cpu();
    void free_memory_all();
};


#endif //DOCKTASK_H
