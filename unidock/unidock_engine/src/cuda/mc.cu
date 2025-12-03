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


__forceinline__ __device__ void randomize_pose_tile(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                    FlexPose* out_pose, const FlexTopo& flex_topo,
                                                    FlexPoseGradient* aux_g, int idx,
                                                    int n, curandStatePhilox4_32_10_t* state){
    Real tmp4[4] = {0};
    float4 rf4 = curand_uniform4(state);

    // generate a random pose inside the box
    if (tile.thread_rank() == 0){
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
            Real a = gyration_radius(out_pose, &flex_topo);
            tmp4[0] = get_real_within(rf4.x, CU_BOX.x_lo + a, CU_BOX.x_hi - a);
            tmp4[1] = get_real_within(rf4.y, CU_BOX.y_lo + a, CU_BOX.y_hi - a);
            tmp4[2] = get_real_within(rf4.z, CU_BOX.z_lo + a, CU_BOX.z_hi - a);

            aux_g->center_g[0] = tmp4[0] - out_pose->center[0];
            aux_g->center_g[1] = tmp4[1] - out_pose->center[1];
            aux_g->center_g[2] = tmp4[2] - out_pose->center[2];

            // random orientation, set gradient.
            // Alexa, M. (2022). Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling of SO(3). Proceedings of the
            // IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2022-June(3), 8281–8290.
            // https://doi.org/10.1109/CVPR52688.2022.00811
            Real s = idx + 0.5;
            Real r = sqrt(s / n);
            Real R = sqrt(1.0 - s / n);
            Real alpha = 2.0 * PI * s / PHI;
            Real beta = 2.0 * PI * s / PSI;
            tmp4[0] = r * sin(alpha);
            tmp4[1] = r * cos(alpha);
            tmp4[2] = R * sin(beta);
            tmp4[3] = R * cos(beta);
            quaternion_to_rotvec(aux_g->orientation_g, tmp4);
            // printf("[RAND] %f, %f, %f\n", aux_g->orientation_g[0], aux_g->orientation_g[1], aux_g->orientation_g[2]);
        }
    }

    rf4 = curand_uniform4(state);
    // generate random uints for all torsions // todo: change sampling of torsions
    for (int i = tile.thread_rank(); i < flex_topo.ntorsion; i += tile.num_threads()){
        // copy dihedrals
        tmp4[3] = out_pose->dihedrals[i];

        // set gradient
        int j = rf4.x * flex_topo.range_inds[i * 2 + 1]; // save index of range_list
        int i_lo = flex_topo.range_inds[i * 2] + j * 2; // save a tmp index

        tmp4[1] = flex_topo.range_list[i_lo + 1] - flex_topo.range_list[i_lo];
        tmp4[0] = get_real_within(rf4.y, flex_topo.range_list[i_lo], flex_topo.range_list[i_lo + 1]);
        aux_g->dihedrals_g[i] = tmp4[0] - tmp4[3];
    }
    tile.sync();

    apply_grad_update_pose_tile(tile, out_pose, aux_g, flex_topo, 1.);
}




__global__ void randomize_pose(FlexPose* out_poses, const FlexTopo* flex_topos, FlexPoseGradient* aux_gradients,
                               curandStatePhilox4_32_10_t* states, int seed,
                               int num_pose_per_flex, int max_thread){
    // randomize all poses by given position & orientation
    // but each with different dihedral

    int id_thread = blockIdx.x * blockDim.x + threadIdx.x; // global idx of the thread
    if (id_thread < max_thread){
        int id_pose = id_thread / TILE_SIZE; // one pose per tile
        int id_flex = id_pose / num_pose_per_flex;

        // Use alias
        FlexPose& out_pose = out_poses[id_pose]; // pointer to global data
        const FlexTopo& flex_topo = flex_topos[id_flex];
        FlexPoseGradient& aux_g = aux_gradients[id_pose * STRIDE_G];

        curandStatePhilox4_32_10_t& state = states[id_pose];

        // Cooperative group
        auto cta = cg::this_thread_block();
        cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(cta);

        // Init curand states
        if (tile.thread_rank() == 0){
            curand_init(seed, id_pose, 0, &state);
        }
        tile.sync();

        int id_pose_per_flex = id_pose % num_pose_per_flex;
        randomize_pose_tile(tile, &out_pose, flex_topo, &aux_g, id_pose_per_flex, num_pose_per_flex, &state);
    }
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
__forceinline__ __device__ void mutate_pose_tile(const cg::thread_block_tile<TILE_SIZE>& tile, FlexPose* out_pose,
                                                 const FlexTopo* flex_topo,
                                                 curandStatePhilox4_32_10_t* state, Real amplitude = 1){
    //amplitude:2.0
    // DOF, which as an index of DOF
    Real rand4[4] = {0};
    Real q[4] = {0};
    Real tmp1[3] = {0};
    Real a = 0;
    int which = -1;

    if (tile.thread_rank() == 0){
        int num_mutable = 2 + flex_topo->ntorsion; //center, orientation, torsions
        if (FLAG_CONSTRAINT_DOCK){
            if (num_mutable < 3){ // no torsion
                which = 3; // so no mutation
            }
            else{
                which = gen_rand_int_within(state, 2, num_mutable - 1);
            }
        }
        else{
            which = gen_rand_int_within(state, 0, num_mutable - 1);
        }
        // DPrint1("which is %d\n", which);
        // prepare random values for choosing DOF to mutate
        gen_4_rand_in_sphere(rand4, state);
    }
    tile.sync();

    which = tile.shfl(which, 0);
    rand4[0] = tile.shfl(rand4[0], 0);
    rand4[1] = tile.shfl(rand4[1], 0);
    rand4[2] = tile.shfl(rand4[2], 0);
    rand4[3] = tile.shfl(rand4[3], 0);

    // 0 for translation
    if (which == 0){
        // compute a translation under box constraint
        tmp1[0] = clamp_by_range(amplitude * rand4[0] + out_pose->center[0], CU_BOX.x_hi, CU_BOX.x_lo) - out_pose->center[0];
        //amplitude * rand_5[0];
        tmp1[1] = clamp_by_range(amplitude * rand4[1] + out_pose->center[1], CU_BOX.y_hi, CU_BOX.y_lo) - out_pose->center[1];
        //amplitude * rand_5[1];
        tmp1[2] = clamp_by_range(amplitude * rand4[2] + out_pose->center[2], CU_BOX.z_hi, CU_BOX.z_lo) - out_pose->center[2];
        //amplitude * rand_5[2];

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
            assert(a > EPSILON);
            // add a random rotation to temporary quaternion
            // the movement step of an atom is roughly amplitude Angstrom
            tmp1[0] = amplitude / a * rand4[0];
            tmp1[1] = amplitude / a * rand4[1];
            tmp1[2] = amplitude / a * rand4[2];

            rotvec_to_quaternion(q, tmp1);

            out_pose->rot_vec[0] = tmp1[0];
            out_pose->rot_vec[1] = tmp1[1];
            out_pose->rot_vec[2] = tmp1[2];
        }
        tile.sync();

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
    else if (which - 2 < flex_topo->ntorsion){
        // rotate one dihedral
        which -= 2;
        // change lig dihedral
        if (tile.thread_rank() == 0){ // todo: check the continuity of gradient
            a = get_real_within(rand4[3], -PI, PI) - out_pose->dihedrals[which];
        }
        a = tile.shfl(a, 0); // increment of dihedral value
        apply_grad_update_dihe_tile(tile, out_pose, flex_topo, which, a);
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
                          curandStatePhilox4_32_10_t* states, int seed,
                          int mc_steps, int opt_steps, int num_pose_per_flex, int max_thread){
    // Just for ONE best pose

    int id_thread = blockIdx.x * blockDim.x + threadIdx.x; // global idx of the thread
    if (id_thread < max_thread){
        int id_pose = id_thread / TILE_SIZE;
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


        duplicate_pose_tile(tile, &pose_accepted, &out_pose, dim, flex_topo.natom);

        if (mc_steps == 0){
            Real energy = cal_e_f_tile(tile, &pose_accepted, flex_topo, fix_mol, flex_param, fix_param, aux_f.f);

            if (tile.thread_rank() == 0){
                pose_accepted.energy = energy;
            }
            duplicate_pose_tile(tile, &out_pose, &pose_accepted, dim, flex_topo.natom);
        }
        else{
            for (int step = 0; step < mc_steps; step++){
                // 1. mutate conf, PRODUCE a random conf
                duplicate_pose_tile(tile, &pose_candidate, &pose_accepted, dim, flex_topo.natom);

                mutate_pose_tile(tile, &pose_candidate, &flex_topo, &state);

                // todo: add clash-detection for efficiency

                // Record initial energy and gradient. E_ori is energy, aux_g is set as current gradient
                // If max_steps == 0, this func only records the energy of original structure.
                if (opt_steps == 0){
                    Real energy = cal_e_grad_tile(tile, &pose_candidate, &aux_g, flex_topo, fix_mol,
                                                  flex_param, fix_param, aux_f.f);

                    if (tile.thread_rank() == 0){
                        pose_candidate.energy = energy;
                    }
                    tile.sync();
                }
                else{
                    // essential optimization. only computes the energy of candidate pose for MC task
                    // coords are updated inside bfgs
                    bfgs_tile(tile,
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
                    duplicate_pose_tile(tile, &pose_accepted, &pose_candidate, dim, flex_topo.natom);

                    // Possibly the best pose by now
                    if (pose_accepted.energy < best_e){
                        duplicate_pose_tile(tile, &out_pose, &pose_accepted, dim, flex_topo.natom);
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

    int npose = nflex * exhuastiveness;
    int nblock = npose * TILE_SIZE / BLOCK_SIZE;
    int nthreads = npose * TILE_SIZE;

    // initilize curand states
    curandStatePhilox4_32_10_t* states;
    checkCUDA(cudaMalloc(&states, sizeof(curandStatePhilox4_32_10_t) * npose));

    // run the kernel
    if (randomize){
        randomize_pose<<<nblock, BLOCK_SIZE>>>(out_poses, topos, aux_gradients,
                                              states, seed,
                                              exhuastiveness, nthreads);
    }

    mc_kernel<<<nblock, BLOCK_SIZE>>>(out_poses, topos, fix_mol,
                                     flex_param, fix_param,
                                     aux_poses, aux_gradients, aux_hessians, aux_forces,
                                     states, seed,
                                     mc_steps, opt_steps, exhuastiveness, nthreads);

    checkCUDA(cudaDeviceSynchronize());
    spdlog::warn("[Line Search Steps Count]: {}", funcCallCount);

    // free mem
    checkCUDA(cudaFree(states));
}
