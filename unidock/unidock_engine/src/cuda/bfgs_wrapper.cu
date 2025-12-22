//
// Created by Congcong Liu on 24-12-9.
//


// src/cuda/bfgs_wrapper.cu
#include "model/model.h"
#include "bfgs.cuh"



__global__ void line_search_one_original_kernel(const FixMol* fix_mol, const FixParamVina* fix_param,
                                   const FlexParamVina* flex_param, const FlexPose* x, const FlexTopo* flex_topo,
                                   FlexPoseGradient* aux_g, FlexPoseGradient* aux_p,
                                   FlexPoseHessian* aux_h, FlexForce* aux_f,
                                   FlexPose* out_x_new, FlexPoseGradient* out_g_new,
                                   Real* out_e, Real* out_alpha) {
    auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());

    Real e0 = cal_e_grad_tile(tile, x, aux_g, *flex_topo, *fix_mol, *flex_param, *fix_param, aux_f->f);
    //DOF, the vector x has this dimension
    int dim_g = 3 + 3 + flex_topo->ntorsion; // center, orientation, torsion
    int dim_x = 3 + 4 + flex_topo->ntorsion;
    // Use unit matrix as initial guess for Hessian
    init_tri_mat_tile(tile, aux_h->matrix, dim_g, 0); // set zero
    set_tri_mat_diagonal_tile(tile, aux_h->matrix, dim_g, 1); // set diagonal to 1

    // compute line search direction aux_p = -Hg (dim*1 vector)
    minus_mat_vec_product_tile(tile, aux_p, aux_h->matrix, aux_g, dim_g);

    // run the target device function
    line_search_tile(tile, *fix_mol, *fix_param,
        *flex_param, x,
        aux_g, *flex_topo, aux_p,
        aux_f->f,
        out_x_new, out_g_new,
        e0, dim_g, dim_x,
        out_e, out_alpha);

}


__global__ void cal_e_grad_one_kernel(const FixMol* fix_mol, const FixParamVina* fix_param,
        const FlexParamVina* flex_param, const FlexPose* x, const FlexTopo* flex_topo,
        FlexPoseGradient* out_g, FlexForce* aux_f, Real* out_e) {

    auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
    *out_e = cal_e_grad_tile(tile, x, out_g, *flex_topo, *fix_mol, *flex_param, *fix_param, aux_f->f);
}


void cal_e_grad_one_gpu(const FixMol& fix_mol, const FixParamVina& fix_param,
        const FlexParamVina& flex_param, const FlexPose& x, const FlexTopo& flex_topo,
        FlexPoseGradient* out_g, Real* out_e) {

    // Allocate memory on GPU
    FixMol* fix_mol_cu = alloccp_FixMol_gpu(fix_mol);
    FixParamVina* fix_param_cu = alloccp_FixParamVina_gpu(fix_param, fix_mol.natom);
    FlexParamVina* flex_param_cu = alloccp_FlexParamVina_gpu(flex_param, flex_topo.natom);
    FlexPose* x_cu = alloccp_FlexPose_gpu(x, flex_topo.natom, flex_topo.ntorsion);
    FlexTopo* flex_topo_cu = alloccp_FlexTopo_gpu(flex_topo);
    FlexPoseGradient* out_g_cu = alloccp_FlexPoseGradient_gpu(*out_g, flex_topo.ntorsion);
    FlexForce* aux_f_cu = alloc_FlexForce_gpu(flex_topo.natom);
    Real* out_e_cu;
    checkCUDA(cudaMalloc(&out_e_cu, sizeof(Real)));

    // Run the kernel
    cal_e_grad_one_kernel<<<1, TILE_SIZE>>>(fix_mol_cu, fix_param_cu, flex_param_cu, x_cu, flex_topo_cu,
        out_g_cu, aux_f_cu, out_e_cu);
    cudaDeviceSynchronize();

    // Copy the results back to CPU
    checkCUDA(cudaMemcpy(out_e, out_e_cu, sizeof(Real), cudaMemcpyDeviceToHost));
    freecp_FlexPoseGradient_gpu(out_g_cu, out_g, flex_topo.ntorsion);

    // Free on GPU
    free_FixMol_gpu(fix_mol_cu);
    free_FixParamVina_gpu(fix_param_cu);
    free_FlexParamVina_gpu(flex_param_cu);
    free_FlexPose_gpu(x_cu);
    free_FlexTopo_gpu(flex_topo_cu);
    free_FlexForce_gpu(aux_f_cu);
    checkCUDA(cudaFree(out_e_cu));

}


/**
 * @brief Wrapper of line search on GPU, only perform on one pose using one warp/block.
 */
void line_search_one_gpu(const FixMol& fix_mol, const FixParamVina& fix_param,
                     const FlexParamVina& flex_param, const FlexPose& x,
                     const FlexTopo& flex_topo,
                     FlexPose* out_x_new, FlexPoseGradient* out_g_new,
                     Real* out_e, Real* out_alpha) {

    // Allocate memory on GPU
    FixMol* fix_mol_cu = alloccp_FixMol_gpu(fix_mol);
    FixParamVina* fix_param_cu = alloccp_FixParamVina_gpu(fix_param, fix_mol.natom);
    FlexParamVina* flex_param_cu = alloccp_FlexParamVina_gpu(flex_param, flex_topo.natom);
    FlexPose* x_cu = alloccp_FlexPose_gpu(x, flex_topo.natom, flex_topo.ntorsion);
    FlexTopo* flex_topo_cu = alloccp_FlexTopo_gpu(flex_topo);
    FlexPoseGradient* aux_g_cu = alloccp_FlexPoseGradient_gpu(*out_g_new, flex_topo.ntorsion);
    FlexPoseGradient* aux_p_cu = alloccp_FlexPoseGradient_gpu(*out_g_new, flex_topo.ntorsion);
    FlexPoseHessian* aux_h_cu = alloc_FlexPoseHessian_gpu(flex_topo.ntorsion);
    FlexForce* aux_f_cu = alloc_FlexForce_gpu(flex_topo.natom);
    FlexPose* out_x_new_cu = alloccp_FlexPose_gpu(*out_x_new, flex_topo.natom, flex_topo.ntorsion);

    FlexPoseGradient* out_g_new_cu = alloccp_FlexPoseGradient_gpu(*out_g_new, flex_topo.ntorsion);
    Real* out_e_cu;
    checkCUDA(cudaMalloc(&out_e_cu, sizeof(Real)));
    Real* out_alpha_cu;
    checkCUDA(cudaMalloc(&out_alpha_cu, sizeof(Real)));


    // Run the kernel
    line_search_one_original_kernel<<<1, TILE_SIZE>>>(fix_mol_cu, fix_param_cu,
        flex_param_cu, x_cu, flex_topo_cu,
        aux_g_cu, aux_p_cu, aux_h_cu, aux_f_cu,
        out_x_new_cu, out_g_new_cu,
        out_e_cu, out_alpha_cu);
    cudaDeviceSynchronize();

    // Copy the results back to CPU
    freecp_FlexPose_gpu(out_x_new_cu, out_x_new, flex_topo.natom, flex_topo.ntorsion);
    freecp_FlexPoseGradient_gpu(out_g_new_cu, out_g_new, flex_topo.ntorsion);

    checkCUDA(cudaMemcpy(out_e, out_e_cu, sizeof(Real), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(out_alpha, out_alpha_cu, sizeof(Real), cudaMemcpyDeviceToHost));

    // Free on GPU
    free_FixMol_gpu(fix_mol_cu);
    free_FixParamVina_gpu(fix_param_cu);
    free_FlexParamVina_gpu(flex_param_cu);
    free_FlexPose_gpu(x_cu);
    free_FlexTopo_gpu(flex_topo_cu);

    checkCUDA(cudaFree(out_e_cu));
    checkCUDA(cudaFree(out_alpha_cu));
    
    free_FlexPoseGradient_gpu(aux_g_cu);
    free_FlexPoseGradient_gpu(aux_p_cu);
    free_FlexPoseHessian_gpu(aux_h_cu);
    free_FlexForce_gpu(aux_f_cu);
}