//
// Created by Congcong Liu on 24-11-21.
//

#include <cooperative_groups.h>
#include "optimize/optimize.h"
#include "model/model.h"
#include "common.cuh"
#include "bfgs.cuh"

namespace cg = cooperative_groups;


__global__ void opt_kernel(FlexPose* out_poses, const int* pose_inds, const FlexTopo* flex_topos, const FixMol& fix_mol,
                          const FlexParamVina* flex_params, const FixParamVina& fix_param,
                          FlexPose* aux_poses, FlexPoseGradient* aux_gradients, FlexPoseHessian* aux_hessians,
                          FlexForce* aux_forces,
                          int refine_steps, int npose_per_flex){

    // nflex, each flex has num_pose poses to optimize, each pose uses a warp

    int id_pose = pose_inds[blockIdx.x];

    int id_flex = id_pose / npose_per_flex;

    // Use alias
    FlexPose& out_pose = out_poses[id_pose]; // pointer to global data
    const FlexTopo& flex_topo = flex_topos[id_flex];
    const FlexParamVina& flex_param = flex_params[id_flex];

    FlexPose& aux_pose_ori = aux_poses[id_pose * STRIDE_POSE + 2];
    FlexPose& aux_pose_new = aux_poses[id_pose * STRIDE_POSE + 3];

    FlexPoseGradient& aux_g = aux_gradients[id_pose * STRIDE_G];
    FlexPoseGradient& aux_g_new = aux_gradients[id_pose * STRIDE_G + 1];
    FlexPoseGradient& aux_g_ori = aux_gradients[id_pose * STRIDE_G + 2];
    FlexPoseGradient& aux_p = aux_gradients[id_pose * STRIDE_G + 3];
    FlexPoseGradient& aux_y = aux_gradients[id_pose * STRIDE_G + 4];
    FlexPoseGradient& aux_minus_hy = aux_gradients[id_pose * STRIDE_G + 5];

    FlexPoseHessian& aux_h = aux_hessians[id_pose];

    FlexForce& aux_f = aux_forces[id_pose];


    // Cooperative group
    auto cta = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(cta);

    bfgs_tile(tile,
               &out_pose, flex_topo, fix_mol, flex_param, fix_param,
               &aux_pose_new, &aux_pose_ori,
               &aux_g, &aux_g_new, &aux_g_ori,
               &aux_p, &aux_y, &aux_minus_hy,
               &aux_h, &aux_f, refine_steps);
    tile.sync();
}



void optimize_cu(FlexPose* out_poses, const int* pose_inds, const FlexTopo* flex_topos, const FixMol& fix_mol,
                          const FlexParamVina* flex_params, const FixParamVina& fix_param,
                          FlexPose* aux_poses, FlexPoseGradient* aux_gradients, FlexPoseHessian* aux_hessians,
                          FlexForce* aux_forces,
                          int refine_steps, int nblock, int npose_per_flex){

    opt_kernel<<<nblock, TILE_SIZE>>>(out_poses, pose_inds, flex_topos, fix_mol,
                          flex_params, fix_param,
                          aux_poses, aux_gradients, aux_hessians,
                          aux_forces, refine_steps, npose_per_flex);
    checkCUDA(cudaDeviceSynchronize());
    spdlog::warn("[Refinement Line Search Steps Count]: {}", funcCallCount);

}
