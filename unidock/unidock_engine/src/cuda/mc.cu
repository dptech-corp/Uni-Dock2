//
// Created by Congcong Liu on 24-9-25.
//

#include <cooperative_groups.h>

#include "myutils/myio.h"
#include "model/model.h"
#include "search/mc.h"

#include "common.cuh"
#include "myutils/errors.h"
#include "myutils/mymath.h"
#include "geometry/quaternion.h"
#include "bfgs.cuh"



__device__ __forceinline__ void randomize_pose_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                    FlexPose* out_pose_new, FlexPoseGradient* aux_g,
                                                    const FlexPose* pose_old, const FlexTopo& flex_topo,
                                                    curandStatePhilox4_32_10_t* state){
    float4 rf4 = curand_uniform4(state);
    Real tmp4[4] = {0};
    Real rotvec[3] = {map_01_to_dot5(rf4.x), map_01_to_dot5(rf4.y), map_01_to_dot5(rf4.z)};
    uint4 ri4 = curand4(state);

    // copy cartesian coordinates
    for (int i = tile.thread_rank(); i < flex_topo.natom * 3; i += tile.num_threads()){
        out_pose_new->coords[i] = pose_old->coords[i];
    }
    tile.sync();

    // generate a random pose inside the box
    if (tile.thread_rank() == 0){
        // set energy
        out_pose_new->energy = pose_old->energy;

        // copy center & orientation
        out_pose_new->center[0] = pose_old->center[0];
        out_pose_new->center[1] = pose_old->center[1];
        out_pose_new->center[2] = pose_old->center[2];
        out_pose_new->rot_vec[0] = pose_old->rot_vec[0];
        out_pose_new->rot_vec[1] = pose_old->rot_vec[1];
        out_pose_new->rot_vec[2] = pose_old->rot_vec[2];

        if (FLAG_CONSTRAINT_DOCK){
            // center and orientation are fixed
            aux_g->center_g[0] = 0;
            aux_g->center_g[1] = 0;
            aux_g->center_g[2] = 0;
            aux_g->orientation_g[0] = 0;
            aux_g->orientation_g[1] = 0;
            aux_g->orientation_g[2] = 0;
        }
        else{
            // random center, set gradient
            Real a = gyration_radius(pose_old, &flex_topo);
            tmp4[0] = get_real_within_by_int(ri4.x, BOX_X_LO + a, BOX_X_HI - a, ceil((BOX_X_HI - BOX_X_LO - 2 * a) / BOX_PREC) + 1);
            tmp4[1] = get_real_within_by_int(ri4.y, BOX_Y_LO + a, BOX_Y_HI - a, ceil((BOX_Y_HI - BOX_Y_LO - 2 * a) / BOX_PREC) + 1);
            tmp4[2] = get_real_within_by_int(ri4.z, BOX_Z_LO + a, BOX_Z_HI - a, ceil((BOX_Z_HI - BOX_Z_LO - 2 * a) / BOX_PREC) + 1);

            aux_g->center_g[0] = tmp4[0] - out_pose_new->center[0];
            aux_g->center_g[1] = tmp4[1] - out_pose_new->center[1];
            aux_g->center_g[2] = tmp4[2] - out_pose_new->center[2];

            // random orientation, set gradient
            rotvec_to_axis_angle(tmp4, rotvec); // tmp4 is axis_angle
            tmp4[0] = rf4.w * PI;
            aux_g->orientation_g[0] = tmp4[1] * tmp4[0];
            aux_g->orientation_g[1] = tmp4[2] * tmp4[0];
            aux_g->orientation_g[2] = tmp4[3] * tmp4[0]; //todo: change to a more efficient sampling
            // if(tile.thread_rank() == 0){
            //     printf("[RAND] %f, %f, %f\n", aux_g->orientation_g[0], aux_g->orientation_g[1], aux_g->orientation_g[2]);
            // }
        }

        // generate random uints for all torsions // todo: change sampling of torsions
        for (int i = 0; i < flex_topo.ntorsion; i ++){
            // copy dihedrals
            tmp4[3] = pose_old->dihedrals[i];

            out_pose_new->dihedrals[i] = tmp4[3];
            // set gradient
            ri4.x = curand(state);
            ri4.w = ri4.x % flex_topo.range_inds[i * 2 + 1]; // save index of range_list
            ri4.z = flex_topo.range_inds[i * 2] + ri4.w * 2; // save a tmp index
            ri4.x = curand(state);

            tmp4[1] = flex_topo.range_list[ri4.z + 1] - flex_topo.range_list[ri4.z];
            ri4.y = ceil(tmp4[1] / TOR_PREC) + 1; // 10 degree as precision

            tmp4[0] = get_real_within_by_int(ri4.x, flex_topo.range_list[ri4.z], flex_topo.range_list[ri4.z + 1], ri4.y);
            aux_g->dihedrals_g[i] = tmp4[0] - tmp4[3];
        }
    }
    tile.sync();

    apply_grad_update_pose_warp(tile, out_pose_new, aux_g, flex_topo, 1.);

}




/**
 * @brief Mutate one pose and update coords.
 * 
 * @param tile Cooperative group
 * @param out_pose Pointer to the pose to be mutated
 * @param flex_topo Topology of the flex
 * @param state cuRand state
 * @param amplitude Amplitude of the mutation
 */
__forceinline__ __device__ void mutate_pose_warp(const cg::thread_block_tile<TILE_SIZE>& tile, FlexPose* out_pose,
                                                 const FlexTopo* flex_topo,
                                                 curandStatePhilox4_32_10_t* state, Real amplitude = 1){ //amplitude:2.0
    // DOF, which as an index of DOF
    Real rand_5[5] = {0};
    Real q[4] = {0}; //todo: use rand_5 instead of q to save registers
    Real tmp1[3] = {0};
    Real a = 0;
    int which = -1;

    if (tile.thread_rank() == 0){
        int num_mutable = 2 + flex_topo->ntorsion; //center, orientation, torsions
        if (FLAG_CONSTRAINT_DOCK){
            if (num_mutable < 3){
                which = 3; // no mutation
            } else{
                which = gen_rand_int_within(state, 2, num_mutable - 1);
            }
        }
        else{
            which = gen_rand_int_within(state, 0, num_mutable - 1);
        }
        // DPrint1("which is %d\n", which);
        // prepare random values for choosing DOF to mutate
        gen_4_rand_in_sphere(rand_5, state);
        rand_5[4] = curand_uniform(state);
    }
    tile.sync();

    which = tile.shfl(which, 0);
    rand_5[0] = tile.shfl(rand_5[0], 0);
    rand_5[1] = tile.shfl(rand_5[1], 0);
    rand_5[2] = tile.shfl(rand_5[2], 0);
    rand_5[3] = tile.shfl(rand_5[3], 0);
    rand_5[4] = tile.shfl(rand_5[4], 0);

    // 0 for translation
    if (which == 0){
        // compute a translation under box constraint
        tmp1[0] = clamp_by_range(amplitude * rand_5[0] + out_pose->center[0], BOX_X_HI, BOX_X_LO) - out_pose->center[0]; //amplitude * rand_5[0];
        tmp1[1] = clamp_by_range(amplitude * rand_5[1] + out_pose->center[1], BOX_Y_HI, BOX_Y_LO) - out_pose->center[1]; //amplitude * rand_5[1];
        tmp1[2] = clamp_by_range(amplitude * rand_5[2] + out_pose->center[2], BOX_Z_HI, BOX_Z_LO) - out_pose->center[2]; //amplitude * rand_5[2];

        for (int i_at = tile.thread_rank(); i_at < flex_topo->natom; i_at += tile.num_threads()){
            // move to the new center
            out_pose->coords[i_at * 3] = out_pose->coords[i_at * 3] + tmp1[0];
            out_pose->coords[i_at * 3 + 1] = out_pose->coords[i_at * 3 + 1] + tmp1[1];
            out_pose->coords[i_at * 3 + 2] = out_pose->coords[i_at * 3 + 2] + tmp1[2];
        }
        tile.sync();
        if (tile.thread_rank() == 0){
            out_pose->center[0] = out_pose->center[0] + tmp1[0];
            out_pose->center[1] = out_pose->center[1] + tmp1[1];
            out_pose->center[2] = out_pose->center[2] + tmp1[2];
        }
        tile.sync();
    }
    else if (which == 1){
        // 1 for rotation
        if (tile.thread_rank() == 0){
            //1 for rotation of the whole molecule
            a = gyration_radius(out_pose, flex_topo); // an indicator of the size
            if (a > EPSILON){
                // add a random rotation to temporary quaternion
                // the movement step of an atom is roughly amplitude Angstrom
                tmp1[0] = amplitude / a * rand_5[0];
                tmp1[1] = amplitude / a * rand_5[1];
                tmp1[2] = amplitude / a * rand_5[2];

                rotvec_to_quaternion(q, tmp1);
                out_pose->rot_vec[0] = tmp1[0];
                out_pose->rot_vec[1] = tmp1[1];
                out_pose->rot_vec[2] = tmp1[2];
            }
        }
        tile.sync();
        a = tile.shfl(a, 0);

        if (a > EPSILON){
            q[0] = tile.shfl(q[0], 0);
            q[1] = tile.shfl(q[1], 0);
            q[2] = tile.shfl(q[2], 0);
            q[3] = tile.shfl(q[3], 0);

            // rotate all atoms fixme: the rotation has low precision and leads to error over 0.001
            for (int i_at = tile.thread_rank(); i_at < flex_topo->natom; i_at += tile.num_threads()){
                tmp1[0] = out_pose->coords[i_at * 3] - out_pose->center[0];
                tmp1[1] = out_pose->coords[i_at * 3 + 1] - out_pose->center[1];
                tmp1[2] = out_pose->coords[i_at * 3 + 2] - out_pose->center[2];
                rotate_vec_by_quaternion(tmp1, q);
                out_pose->coords[i_at * 3] = tmp1[0] + out_pose->center[0];
                out_pose->coords[i_at * 3 + 1] = tmp1[1] + out_pose->center[1];
                out_pose->coords[i_at * 3 + 2] = tmp1[2] + out_pose->center[2];
            }
            tile.sync();
        }
    }
    else if (which - 2 < flex_topo->ntorsion){
        // rotate one dihedral
        which -= 2;
        // change lig dihedral
        if (tile.thread_rank() == 0){
            a = get_radian_in_ranges(flex_topo->range_list + flex_topo->range_inds[2 * which],
                                     flex_topo->range_inds[2 * which + 1], rand_5 + 3) - out_pose->dihedrals[which];
            // printf("which is %d, a is %f\n", which,  a);
        }
        a = tile.shfl(a, 0); // increment of dihedral value
        apply_grad_update_dihe_warp(tile, out_pose, flex_topo, which, a);
    }
    else{
        // no mutation
        // assert(which - 2 < flex_topo->ntorsion);
    }
}


/**
 * @brief Monte Carlo Kernel. Each warp tackles one pose.
 *
 * @param out_poses Prepared poses that have been initialized, size: nflex * num_pose_per_flex
 * @param flex_topos Topology of flex, size: nflex
 * @param aux_poses Auxiliary poses, size: STRIDE_POSE * nflex * num_pose_per_flex
 * @param aux_gradients Auxiliary gradients, size: STRIDE_G * nflex * num_pose_per_flex
 * @param aux_hessians Auxiliary hessians, size: nflex * num_pose_per_flex
 * @param aux_forces Auxiliary forces, size: nflex * num_pose_per_flex
 * @param states Random states, each pose owns one, size: nflex * num_pose_per_flex
 * @param seed Random seed
 * @param mc_steps Number of MC steps
 * @param opt_steps Number of optimization steps after a pose is accepted
 * @param num_pose_per_flex Number of poses per flex
 * @param max_thread Maximum number of threads used
 */
__global__ void mc_kernel(FlexPose* out_poses, const FlexTopo* flex_topos, const FixMol& fix_mol,
                          const FlexParamVina* flex_params, const FixParamVina& fix_param,
                          FlexPose* aux_poses, FlexPoseGradient* aux_gradients, FlexPoseHessian* aux_hessians,
                          FlexForce* aux_forces,
                          curandStatePhilox4_32_10_t* states, int seed, bool randomize,
                          int mc_steps, int opt_steps, int num_pose_per_flex, int max_thread){
    // Just for ONE best pose

    int id_thread = blockIdx.x * blockDim.x + threadIdx.x; // global idx of the thread
    if (id_thread < max_thread){
        int id_pose = id_thread / TILE_SIZE; // 一个pose由1个tile处理，也就是包含多个threads
        int id_flex = id_pose / num_pose_per_flex;

        // Use alias
        FlexPose& out_pose = out_poses[id_pose]; // pointer to global data
        const FlexTopo& flex_topo = flex_topos[id_flex];
        const FlexParamVina& flex_param = flex_params[id_flex];

        FlexPose& pose_candidate = aux_poses[id_pose * STRIDE_POSE];
        FlexPose& pose_accepted = aux_poses[id_pose * STRIDE_POSE + 1];
        FlexPose& aux_pose_ori = aux_poses[id_pose * STRIDE_POSE + 2];
        FlexPose& aux_pose_new = aux_poses[id_pose * STRIDE_POSE + 3];

        FlexPoseGradient& aux_g = aux_gradients[id_pose * STRIDE_G];
        FlexPoseGradient& aux_g_new = aux_gradients[id_pose * STRIDE_G + 1];
        FlexPoseGradient& aux_g_ori = aux_gradients[id_pose * STRIDE_G + 2];
        FlexPoseGradient& aux_p = aux_gradients[id_pose * STRIDE_G + 3];
        FlexPoseGradient& aux_y = aux_gradients[id_pose * STRIDE_G + 4];
        FlexPoseGradient& aux_minus_hy = aux_gradients[id_pose * STRIDE_G + 5];

        FlexPoseHessian& aux_h = aux_hessians[id_pose];

        FlexForce& aux_f = aux_forces[id_pose]; //todo: use struct or just Real* ?

        curandStatePhilox4_32_10_t& state = states[id_pose];


        // Cooperative group
        auto cta = cg::this_thread_block();
        cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(cta);
        // Init curand states
        if (tile.thread_rank() == 0){
            curand_init(seed, id_pose, 0, &state);
        }
        tile.sync();

        int dim = 3 + 4 + flex_topo.ntorsion;
        Real best_e = 1e9; // large value for finding minimum energy

        if (randomize){
            // prepare the initial pose: each pose is a random pose!
            randomize_pose_warp(tile, &pose_accepted, &aux_g, &out_pose, flex_topo, &state);
        }else{
            duplicate_pose_warp(tile, &pose_accepted, &out_pose, dim, flex_topo.natom);
        }

        if (mc_steps == 0){
            Real energy = cal_e_f_warp(tile, &pose_accepted, flex_topo, fix_mol, flex_param, fix_param, aux_f.f);

            if (tile.thread_rank() == 0){
                pose_accepted.energy = energy;
            }
            duplicate_pose_warp(tile, &out_pose, &pose_accepted, dim, flex_topo.natom);
        }
        else{
            for (int step = 0; step < mc_steps; step++){
                // 1. mutate conf, PRODUCE a random conf
                DPrint1("========= MC step %d \n", step);

                duplicate_pose_warp(tile, &pose_candidate, &pose_accepted, dim, flex_topo.natom);

                mutate_pose_warp(tile, &pose_candidate, &flex_topo, &state);

                // todo: add clash-detection for efficiency

                // Record initial energy and gradient. E_ori is energy, aux_g is set as current gradient
                // If max_steps == 0, this func only records the energy of original structure.
                if (opt_steps == 0){
                    Real energy = cal_e_grad_warp(tile, &pose_candidate, &aux_g, flex_topo, fix_mol,
                                                  flex_param, fix_param, aux_f.f);

                    if (tile.thread_rank() == 0){
                        pose_candidate.energy = energy;
                    }
                    tile.sync();
                }
                else{
                    // essential optimization. only computes the energy of candidate pose for MC task
                    // coords are updated inside bfgs
                    bfgs_warp(tile,
                              &pose_candidate, flex_topo, fix_mol,
                              flex_param, fix_param,
                              &aux_pose_new, &aux_pose_ori,
                              &aux_g, &aux_g_new, &aux_g_ori,
                              &aux_p, &aux_y, &aux_minus_hy,
                              &aux_h, &aux_f, opt_steps);
                }


                // 2. Metropolis
                bool accepted = false;
                if (tile.thread_rank() == 0){
                    Real rand_x = curand_uniform(&states[id_pose]);
                    accepted = metropolis_accept(pose_accepted.energy, pose_candidate.energy, 1.2, rand_x);
                }
                accepted = tile.shfl(accepted, 0);

                // if accepted
                if (step == 0 || accepted){
                    // set accepted pose as this lately accepted candidate
                    duplicate_pose_warp(tile, &pose_accepted, &pose_candidate, dim, flex_topo.natom);

                    // Possibly the best pose by now
                    if (pose_accepted.energy < best_e){
                        duplicate_pose_warp(tile, &out_pose, &pose_accepted, dim, flex_topo.natom);
                        best_e = pose_accepted.energy;
                    }
                }
            }
        }
    }
}


void mc_cu(FlexPose* out_poses, const FlexTopo* topos,
           const FixMol& fix_mol, const FlexParamVina* flex_param, const FixParamVina& fix_param,
           FlexPose* aux_poses, FlexPoseGradient* aux_gradients, FlexPoseHessian* aux_hessians, FlexForce* aux_forces,
           int mc_steps, int opt_steps, int nflex, int exhuastiveness, int seed, bool randomize){
    //------- perform MC on GPU -------//

    const int block_size = TILE_SIZE; // One block for one tile (for 32, namely one warp per block)
    int npose = nflex * exhuastiveness;

    // initilize curand states
    curandStatePhilox4_32_10_t* states;
    checkCUDA(cudaMalloc(&states, sizeof(curandStatePhilox4_32_10_t) * npose));

    // run the kernel
    mc_kernel<<<npose, block_size>>>(out_poses, topos, fix_mol,
                                     flex_param, fix_param,
                                     aux_poses, aux_gradients, aux_hessians, aux_forces,
                                     states, seed, randomize,
                                     mc_steps, opt_steps, exhuastiveness, npose * block_size);
    checkCUDA(cudaDeviceSynchronize());
    spdlog::warn("[Line Search Steps Count]: {}", funcCallCount);

    // free mem
    checkCUDA(cudaFree(states));
}
