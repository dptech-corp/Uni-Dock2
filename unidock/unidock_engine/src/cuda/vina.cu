//
// Created by Congcong Liu on 24-12-9.
//


#include "score/vina.h"

// ==================  Memory allocation for ONE object ==================
FlexParamVina* alloccp_FlexParamVina_gpu(const FlexParamVina& flex_param_vina, int natom){
    FlexParamVina* flex_param_vina_cu;
    cudaMalloc(&flex_param_vina_cu, sizeof(FlexParamVina));

    FlexParamVina flex_param_vina_tmp;

    // Compute total intra adjacency list size from intra_range
    int total_intra = 0;
    if (natom > 0) {
        total_intra = flex_param_vina.intra_range[(natom - 1) * 2] + flex_param_vina.intra_range[(natom - 1) * 2 + 1];
    }
    cudaMalloc(&flex_param_vina_tmp.pairs_intra, sizeof(int) * total_intra);
    cudaMemcpy(flex_param_vina_tmp.pairs_intra, flex_param_vina.pairs_intra, sizeof(int) * total_intra, cudaMemcpyHostToDevice);
    cudaMalloc(&flex_param_vina_tmp.intra_range, sizeof(int) * natom * 2);
    cudaMemcpy(flex_param_vina_tmp.intra_range, flex_param_vina.intra_range, sizeof(int) * natom * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&flex_param_vina_tmp.atom_types, sizeof(int) * natom);
    cudaMemcpy(flex_param_vina_tmp.atom_types, flex_param_vina.atom_types, sizeof(int) * natom, cudaMemcpyHostToDevice);

    // Handle bias fields
    cudaMalloc(&flex_param_vina_tmp.inds_bias, sizeof(int) * natom * 2);
    cudaMemcpy(flex_param_vina_tmp.inds_bias, flex_param_vina.inds_bias, sizeof(int) * natom * 2, cudaMemcpyHostToDevice);
    
    // params_bias can be nullptr if no bias data exists
    if (flex_param_vina.params_bias != nullptr) {
        // Calculate size: count non-zero bias ranges
        int nbias_params = 0;
        for (int i = 0; i < natom; i++) {
            int end_idx = flex_param_vina.inds_bias[i * 2 + 1];
            if (end_idx > nbias_params) {
                nbias_params = end_idx;
            }
        }
        if (nbias_params > 0) {
            cudaMalloc(&flex_param_vina_tmp.params_bias, sizeof(Real) * nbias_params);
            cudaMemcpy(flex_param_vina_tmp.params_bias, flex_param_vina.params_bias, sizeof(Real) * nbias_params, cudaMemcpyHostToDevice);
        } else {
            flex_param_vina_tmp.params_bias = nullptr;
        }
    } else {
        flex_param_vina_tmp.params_bias = nullptr;
    }

    cudaMemcpy(flex_param_vina_cu, &flex_param_vina_tmp, sizeof(FlexParamVina), cudaMemcpyHostToDevice);
    return flex_param_vina_cu;
}
void free_FlexParamVina_gpu(FlexParamVina* flex_param_vina_cu){
    FlexParamVina flex_param_vina_tmp;
    cudaMemcpy(&flex_param_vina_tmp, flex_param_vina_cu, sizeof(FlexParamVina), cudaMemcpyDeviceToHost);
    cudaFree(flex_param_vina_tmp.atom_types);
    cudaFree(flex_param_vina_tmp.pairs_intra);
    cudaFree(flex_param_vina_tmp.intra_range);
    cudaFree(flex_param_vina_tmp.inds_bias);
    if (flex_param_vina_tmp.params_bias != nullptr) {
        cudaFree(flex_param_vina_tmp.params_bias);
    }
    cudaFree(flex_param_vina_cu);
}


FixParamVina* alloccp_FixParamVina_gpu(const FixParamVina& fix_param_vina, int natom){
    FixParamVina* fix_param_vina_cu;
    cudaMalloc(&fix_param_vina_cu, sizeof(FixParamVina));

    FixParamVina fix_param_vina_tmp;
    cudaMalloc(&fix_param_vina_tmp.atom_types, sizeof(int) * natom);
    cudaMemcpy(fix_param_vina_tmp.atom_types, fix_param_vina.atom_types, sizeof(int) * natom, cudaMemcpyHostToDevice);

    cudaMemcpy(fix_param_vina_cu, &fix_param_vina_tmp, sizeof(FixParamVina), cudaMemcpyHostToDevice);
    return fix_param_vina_cu;
}
void free_FixParamVina_gpu(FixParamVina* fix_param_vina_cu){
    FixParamVina fix_param_vina_tmp;
    cudaMemcpy(&fix_param_vina_tmp, fix_param_vina_cu, sizeof(FixParamVina), cudaMemcpyDeviceToHost);
    cudaFree(fix_param_vina_tmp.atom_types);
    cudaFree(fix_param_vina_cu);
}
