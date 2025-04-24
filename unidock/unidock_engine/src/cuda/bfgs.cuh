//
// Created by Congcong Liu on 24-10-23.
//

#ifndef BFGS_CUH
#define BFGS_CUH

#include <cooperative_groups/reduce.h>
#include "common.cuh"
#include "mymath_warp.cuh"
#include "geometry/quaternion.h"

namespace cg = cooperative_groups;


__device__ __forceinline__ void duplicate_grad_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                    FlexPoseGradient* out_g_new, const FlexPoseGradient* g_old,
                                                    int dim){
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()){
        if (i < 3){
            out_g_new->center_g[i] = g_old->center_g[i];
        }
        else if (i < dof_g){
            out_g_new->orientation_g[i - 3] = g_old->orientation_g[i - 3];
        }
        else{
            out_g_new->dihedrals_g[i - dof_g] = g_old->dihedrals_g[i - dof_g];
        }
    }
    tile.sync();
}


__device__ __forceinline__ void duplicate_pose_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                    FlexPose* out_pose_new, const FlexPose* pose_old, int dim,
                                                    int natom){
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()){
        if (i < 3){
            out_pose_new->center[i] = pose_old->center[i];
        }
        else if (i < dof_x){
            out_pose_new->orientation[i - 3] = pose_old->orientation[i - 3];
        }
        else{
            out_pose_new->dihedrals[i - dof_x] = pose_old->dihedrals[i - dof_x];
        }
    }
    tile.sync();

    // copy cartesian coordinates
    for (int i = tile.thread_rank(); i < natom * 3; i += tile.num_threads()){
        out_pose_new->coords[i] = pose_old->coords[i];
    }
    tile.sync();

    // set energy
    if (tile.thread_rank() == 0){
        // assign e but not coords
        out_pose_new->energy = pose_old->energy;
    }
    tile.sync();
}


/**
 * @brief Compute gradient on dihedral, center, and orientation
 */
__device__ __forceinline__ void cal_grad_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                               const Real* aux_f, const FlexTopo& flex_topo,
                                               const FlexPose* pose, FlexPoseGradient* out_g
){
    Real tmp1[3] = {0}, tmp2[3] = {0}, tmp3[3] = {0}, tmp4[3] = {0}, torque[3] = {0};
    int i_at = 0;

    // each thread for a torsion. Cal gradient on dihedral
    for (int i_tor = tile.thread_rank(); i_tor < flex_topo.ntorsion; i_tor += tile.num_threads()){
        int begin = flex_topo.rotated_inds[i_tor * 2];
        int end = flex_topo.rotated_inds[i_tor * 2 + 1];
        int iat_from = flex_topo.axis_atoms[i_tor * 2];
        int iat_to = flex_topo.axis_atoms[i_tor * 2 + 1];

        torque[0] = torque[1] = torque[2] = 0.;

        for (int j = begin; j < end + 1; j++){
            i_at = flex_topo.rotated_atoms[j];
            if (flex_topo.vn_types[i_at] == VN_TYPE_H){
                continue;
            }
            // compute the lever arm vector, "to" atom is the origin of this fragment
            tmp1[0] = pose->coords[i_at * 3] - pose->coords[iat_to * 3];
            tmp1[1] = pose->coords[i_at * 3 + 1] - pose->coords[iat_to * 3 + 1];
            tmp1[2] = pose->coords[i_at * 3 + 2] - pose->coords[iat_to * 3 + 2];

            // take gradient from aux_f
            tmp2[0] = aux_f[i_at * 3];
            tmp2[1] = aux_f[i_at * 3 + 1];
            tmp2[2] = aux_f[i_at * 3 + 2];

            // compute the torque on the atom for the torsion
            cross_product(tmp1, tmp2, tmp3);

            // accumulate the torque for the torsion
            torque[0] += tmp3[0];
            torque[1] += tmp3[1];
            torque[2] += tmp3[2];
        }

        // take the axis
        tmp3[0] = pose->coords[iat_to * 3] - pose->coords[iat_from * 3];
        tmp3[1] = pose->coords[iat_to * 3 + 1] - pose->coords[iat_from * 3 + 1];
        tmp3[2] = pose->coords[iat_to * 3 + 2] - pose->coords[iat_from * 3 + 2];

        // projection on axis as the gradient on dihedral
        out_g->dihedrals_g[i_tor] = dot_product(torque, tmp3) / norm_vec3(tmp3);
    }
    tile.sync();

    // Derivative on center
    torque[0] = torque[1] = torque[2] = 0.;
    for (i_at = tile.thread_rank(); i_at < flex_topo.natom; i_at += tile.num_threads()){
        if (flex_topo.vn_types[i_at] == VN_TYPE_H){
            continue;
        }
        // compute the lever arm vector to center
        tmp1[0] = pose->coords[i_at * 3] - pose->center[0];
        tmp1[1] = pose->coords[i_at * 3 + 1] - pose->center[1];
        tmp1[2] = pose->coords[i_at * 3 + 2] - pose->center[2];

        // take force grad from aux_f
        tmp2[0] = aux_f[i_at * 3];
        tmp2[1] = aux_f[i_at * 3 + 1];
        tmp2[2] = aux_f[i_at * 3 + 2];

        tmp4[0] += tmp2[0];
        tmp4[1] += tmp2[1];
        tmp4[2] += tmp2[2];

        // compute the torque on the atom for the center
        cross_product(tmp1, tmp2, tmp3);
        torque[0] += tmp3[0];
        torque[1] += tmp3[1];
        torque[2] += tmp3[2];

        DPrint("Atom %d V: %f, %f, %f F: %f, %f, %f Torque: %f, %f, %f\n",
            i_at, tmp1[0], tmp1[1], tmp1[2], tmp2[0], tmp2[1], tmp2[2],
            tmp3[0], tmp3[1], tmp3[2]);
    }
    tile.sync();
    // -force
    tmp1[0] = cg::reduce(tile, tmp4[0], cg::plus<Real>());
    tmp1[1] = cg::reduce(tile, tmp4[1], cg::plus<Real>());
    tmp1[2] = cg::reduce(tile, tmp4[2], cg::plus<Real>());
    // -torque
    torque[0] = cg::reduce(tile, torque[0], cg::plus<Real>());
    torque[1] = cg::reduce(tile, torque[1], cg::plus<Real>());
    torque[2] = cg::reduce(tile, torque[2], cg::plus<Real>());
    tile.sync();


    // write gradient of center
    if (tile.thread_rank() == 0){
        out_g->center_g[0] = tmp1[0];
        out_g->center_g[1] = tmp1[1];
        out_g->center_g[2] = tmp1[2];
        DPrint("Total force: %f, %f, %f\n", tmp1[0], tmp1[1], tmp1[2]);

        // projection on random axis as the gradient on orientation
        // todo: incorrect but may work
        out_g->orientation_g[0] = torque[0];
        out_g->orientation_g[1] = torque[1];
        out_g->orientation_g[2] = torque[2];
        out_g->orientation_g[3] = 0;
        DPrint("Total Torque: %f, %f, %f\n", torque[0], torque[1], torque[2]);
    }
    tile.sync();
}


SCOPE_INLINE Real cal_box_penalty_clamp_coord(Real* out_x, Real* out_y, Real* out_z, Real* out_f){
    Real penalty = 0.;
    if (*out_x < BOX_X_LO){
        penalty += BOX_X_LO - *out_x;
        *out_x = BOX_X_LO;
        out_f[0] += -PENALTY_SLOPE;
    }
    else if (*out_x > BOX_X_HI){
        penalty += *out_x - BOX_X_HI;
        *out_x = BOX_X_HI;
        out_f[0] -= -PENALTY_SLOPE;
    }

    if (*out_y < BOX_Y_LO){
        penalty += BOX_Y_LO - *out_y;
        *out_y = BOX_Y_LO;
        out_f[1] += -PENALTY_SLOPE;
    }
    else if (*out_y > BOX_Y_HI){
        penalty += *out_y - BOX_Y_HI;
        *out_y = BOX_Y_HI;
        out_f[1] -= -PENALTY_SLOPE;
    }

    if (*out_z < BOX_Z_LO){
        penalty += BOX_Z_LO - *out_z;
        *out_z = BOX_Z_LO;
        out_f[2] += -PENALTY_SLOPE;
    }
    else if (*out_z > BOX_Z_HI){
        penalty += *out_z - BOX_Z_HI;
        *out_z = BOX_Z_HI;
        out_f[2] -= -PENALTY_SLOPE;
    }

    return penalty * PENALTY_SLOPE;
}



__device__ __forceinline__ Real cal_e_f_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                             const FlexPose* pose, const FlexTopo& flex_topo,
                                             const FixMol& fix_mol, const FlexParamVina& flex_param,
                                             const FixParamVina& fix_param,
                                             Real* aux_f){
    Real energy = 0.;
    Real rr = 0.;
    Real f_div_r = 0.;
    Real f_tmp[3] = {0.};
    Real coord_adj[3] = {0.};


    // 0. initialize aux_f as zero
    for (int i = tile.thread_rank(); i < flex_topo.natom * 3; i += tile.num_threads()){
        aux_f[i] = 0.;
    }
    tile.sync();


    // 1. Compute Pairwise energy and forces
    // -- Compute inter-molecular energy: flex-protein
    for (int i = tile.thread_rank(); i < flex_param.npair_inter; i += tile.num_threads()){
        int i1 = flex_param.pairs_inter[i * 2], i2 = flex_param.pairs_inter[i * 2 + 1];
        assert(i1 < flex_topo.natom && i2 < fix_mol.natom);

        // check each i1's penalty and modify the coords if necessary
        coord_adj[0] = pose->coords[i1 * 3];
        coord_adj[1] = pose->coords[i1 * 3 + 1];
        coord_adj[2] = pose->coords[i1 * 3 + 2];
        cal_box_penalty_clamp_coord(coord_adj, coord_adj + 1, coord_adj + 2, f_tmp);

        // Cartesian distances won't be saved
        Real r_vec[3] = {
            fix_mol.coords[i2 * 3] -  coord_adj[0],
            fix_mol.coords[i2 * 3 + 1] -  coord_adj[1],
            fix_mol.coords[i2 * 3 + 2] -  coord_adj[2]
        };
        rr = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];

        if (rr < Score.r2_cutoff){
            rr = sqrt(rr); // use r2 as a container for |r|
            Real e_tmp = Score.eval_ef(rr - flex_param.r1_plus_r2_inter[i], flex_param.atom_types[i1],
                                    fix_param.atom_types[i2], &f_div_r);
            energy += e_tmp;

            if (rr < EPSILON){ // robust
                rr = EPSILON;
                // CUDA_ERROR("[INTER] Two atoms overlap! i1: %d, i2: %d, r_vec:%f, %f, %f, f: %.10f\n",
                //            i1, i2, r_vec[0], r_vec[1], r_vec[2], f_div_r);
            }
            f_div_r /= rr; //now it is f / |r|

            atomicAdd_block(aux_f + i1 * 3, -f_div_r * r_vec[0]);
            atomicAdd_block(aux_f + i1 * 3 + 1, -f_div_r * r_vec[1]);
            atomicAdd_block(aux_f + i1 * 3 + 2, -f_div_r * r_vec[2]);
        }
    }
    tile.sync();


    // -- Compute inter-molecular energy: penalty
    for (int i = tile.thread_rank(); i < flex_topo.natom; i += tile.num_threads()){
        f_tmp[0] = 0.;
        f_tmp[1] = 0.;
        f_tmp[2] = 0.;
        if (flex_param.atom_types[i] != VN_TYPE_H){
            coord_adj[0] = pose->coords[i * 3];
            coord_adj[1] = pose->coords[i * 3 + 1];
            coord_adj[2] = pose->coords[i * 3 + 2];
            energy += cal_box_penalty_clamp_coord(coord_adj, coord_adj + 1, coord_adj + 2, f_tmp);

            if (abs(f_tmp[0]) > EPSILON){
                aux_f[i * 3] += f_tmp[0];
            }
            if (abs(f_tmp[1]) > EPSILON){
                aux_f[i * 3 + 1] += f_tmp[1];
            }
            if (abs(f_tmp[2]) > EPSILON){
                aux_f[i * 3 + 2] += f_tmp[2];
            }
        }
    }
    tile.sync();


    // -- Compute intra-molecular energy
    for (int i = tile.thread_rank(); i < flex_param.npair_intra; i += tile.num_threads()){
        int i1 = flex_param.pairs_intra[i * 2], i2 = flex_param.pairs_intra[i * 2 + 1];
        // DPrint1("i1:%d, i2:%d\n", i1, i2);
        // Cartesian distances won't be saved
        Real r_vec[3] = {
            pose->coords[i2 * 3] - pose->coords[i1 * 3],
            pose->coords[i2 * 3 + 1] - pose->coords[i1 * 3 + 1],
            pose->coords[i2 * 3 + 2] - pose->coords[i1 * 3 + 2]
        };
        rr = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];

        if (rr < Score.r2_cutoff){
            rr = sqrt(rr); // use r2 as a container for |r|
            energy += Score.eval_ef(rr - flex_param.r1_plus_r2_intra[i], flex_param.atom_types[i1],
                                    flex_param.atom_types[i2], &f_div_r);
            if (rr < EPSILON){// robust
                rr = EPSILON;
                // CUDA_ERROR("[INTRA] Two atoms overlap! i1: %d, i2: %d, r_vec:%f, %f, %f, f: %.10f\n",
                //            i1, i2, r_vec[0], r_vec[1], r_vec[2], f_div_r);
            }
            f_div_r /= rr; // till now, it is the real f/|r|
            assert(i1 < flex_topo.natom && i2 < flex_topo.natom);
            atomicAdd_block(aux_f + i1 * 3, -f_div_r * r_vec[0]);
            atomicAdd_block(aux_f + i1 * 3 + 1, -f_div_r * r_vec[1]);
            atomicAdd_block(aux_f + i1 * 3 + 2, -f_div_r * r_vec[2]);
            atomicAdd_block(aux_f + i2 * 3, f_div_r * r_vec[0]);
            atomicAdd_block(aux_f + i2 * 3 + 1, f_div_r * r_vec[1]);
            atomicAdd_block(aux_f + i2 * 3 + 2, f_div_r * r_vec[2]);
        }
    }
    tile.sync();

    // 2. Compute energy and forces by dihedral
    //bla bla...

    // Sum the total energy of all threads in this warp
    energy = cg::reduce(tile, energy, cg::plus<Real>());

    return energy;
}


__device__ __forceinline__ Real cal_e_grad_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                const FlexPose* pose, FlexPoseGradient* out_g,
                                                const FlexTopo& flex_topo,
                                                const FixMol& fix_mol, const FlexParamVina& flex_param,
                                                const FixParamVina& fix_param,
                                                Real* aux_f){
    // Total energy
    Real energy = cal_e_f_warp(tile, pose, flex_topo, fix_mol, flex_param, fix_param, aux_f);

    // Compute gradients on -force
    cal_grad_warp(tile, aux_f, flex_topo, pose, out_g);
    return energy;
}


__device__ __forceinline__ void minus_mat_vec_product_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                           FlexPoseGradient* out_p,
                                                           const Real* h, const FlexPoseGradient* g,
                                                           int dim){
    // return -Hg
    // here uses square matrix dim * dim
    Real sum = 0;
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()){
        sum = 0;
        for (int j = 0; j < dim; j++){
            sum += h[tri_mat_index(i, j)] * grad_index_read(g, j);
        }
        grad_index_write(out_p, i, -sum);
    }
    tile.sync();
}

__device__ __forceinline__ Real g_dot_product_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                   const FlexPoseGradient* a, const FlexPoseGradient* b,
                                                   int dim){
    Real tmp = 0;
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()){
        tmp += grad_index_read(a, i) * grad_index_read(b, i);
    }
    tile.sync();

    return cg::reduce(tile, tmp, cg::plus<Real>());
}


__device__ __forceinline__ void apply_grad_update_dihe_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                            FlexPose* out_x, const FlexTopo* flex_topo,
                                                            int i_tor, Real dihe_incre_raw){
    int begin = 0;
    int end = 0;
    Real tmp1[3] = {0};
    Real coord_to[3] = {0};
    Real axis_unit[3] = {0};
    Real dihe_incre = 0;
    int iat_from = -1;
    int iat_to = -1;

    if (tile.thread_rank() == 0){
        dihe_incre = normalize_angle(dihe_incre_raw);
        // DPrint1("i_tor: %d, dihe_incre_raw: %f, normalized dihe_incre: %f\n", i_tor, dihe_incre_raw, dihe_incre);

        // apply constraint by Torsion Library
        int i_lo = flex_topo->range_inds[i_tor * 2];
        tmp1[0] = normalize_angle(out_x->dihedrals[i_tor] + dihe_incre);
        Real dihe_new = clamp_by_ranges(tmp1[0],
                                        flex_topo->range_list + i_lo, flex_topo->range_inds[i_tor * 2 + 1]);
        // DPrint("Ranges: %f, %f, %f, %f, %f, %f\n\n",
        //     flex_topo->range_list[0], flex_topo->range_list[1], flex_topo->range_list[2], flex_topo->range_list[3],
        //     flex_topo->range_list[4], flex_topo->range_list[5]);

        // update dihedral value
        dihe_incre = dihe_new - out_x->dihedrals[i_tor];
        // DPrint("dihe_new[%d] after clamping: %f, dihe_now: %f, dihe_incre: %f\n", i_tor, dihe_new, out_x->dihedrals[i_tor], dihe_incre);
        out_x->dihedrals[i_tor] = dihe_new;

        // influenced atoms
        begin = flex_topo->rotated_inds[i_tor * 2];
        end = flex_topo->rotated_inds[i_tor * 2 + 1];

        // define rotation axis
        iat_from = flex_topo->axis_atoms[i_tor * 2];
        iat_to = flex_topo->axis_atoms[i_tor * 2 + 1];
        coord_to[0] = out_x->coords[iat_to * 3];
        coord_to[1] = out_x->coords[iat_to * 3 + 1];
        coord_to[2] = out_x->coords[iat_to * 3 + 2];

        axis_unit[0] = coord_to[0] - out_x->coords[iat_from * 3];
        axis_unit[1] = coord_to[1] - out_x->coords[iat_from * 3 + 1];
        axis_unit[2] = coord_to[2] - out_x->coords[iat_from * 3 + 2];

        Real l = norm_vec3(axis_unit);

        if (l <= EPSILON_cu){ //todo: not allowed
            // CUDA_ERROR("l <= EPSILON_cu");
            axis_unit[0] = 1;
            axis_unit[1] = 0;
            axis_unit[2] = 0;
        }
        else{
            axis_unit[0] /= l;
            axis_unit[1] /= l;
            axis_unit[2] /= l;
        }
    }
    tile.sync();
    begin = tile.shfl(begin, 0);
    end = tile.shfl(end, 0);
    coord_to[0] = tile.shfl(coord_to[0], 0);
    coord_to[1] = tile.shfl(coord_to[1], 0);
    coord_to[2] = tile.shfl(coord_to[2], 0);
    axis_unit[0] = tile.shfl(axis_unit[0], 0);
    axis_unit[1] = tile.shfl(axis_unit[1], 0);
    axis_unit[2] = tile.shfl(axis_unit[2], 0);
    dihe_incre = tile.shfl(dihe_incre, 0);


    // parallel on updating atomic coords
    for (int j = tile.thread_rank(); j < end - begin + 1; j += tile.num_threads()){
        int i_at = flex_topo->rotated_atoms[j + begin];
        // move the rotation origin to the axis
        tmp1[0] = out_x->coords[i_at * 3] - coord_to[0];
        tmp1[1] = out_x->coords[i_at * 3 + 1] - coord_to[1];
        tmp1[2] = out_x->coords[i_at * 3 + 2] - coord_to[2];
        // Rotate atom by Rodrigues' formula
        rotate_vec_by_rodrigues(tmp1, axis_unit, dihe_incre);
        // move back to the original position
        out_x->coords[i_at * 3] = tmp1[0] + coord_to[0];
        out_x->coords[i_at * 3 + 1] = tmp1[1] + coord_to[1];
        out_x->coords[i_at * 3 + 2] = tmp1[2] + coord_to[2];
    }
    tile.sync();

    // update center
    Real center_new[3] = {0};
    for (int j = tile.thread_rank(); j < flex_topo->natom; j += tile.num_threads()){
        center_new[0] += out_x->coords[j * 3];
        center_new[1] += out_x->coords[j * 3 + 1];
        center_new[2] += out_x->coords[j * 3 + 2];
    }
    tile.sync();

    center_new[0] = cg::reduce(tile, center_new[0], cg::plus<Real>());
    center_new[1] = cg::reduce(tile, center_new[1], cg::plus<Real>());
    center_new[2] = cg::reduce(tile, center_new[2], cg::plus<Real>());

    if (tile.thread_rank() == 0){
        out_x->center[0] = center_new[0] / flex_topo->natom;
        out_x->center[1] = center_new[1] / flex_topo->natom;
        out_x->center[2] = center_new[2] / flex_topo->natom;
    }
    tile.sync();
}


/**
 * @brief Apply gradient to update the pose, including center, orientation, and dihedrals
 * @param tile Thread block tile
 * @param out_x Pose to be updated
 * @param g Gradient
 * @param flex_topo Flex topology
 * @param alpha Step length
 */
__device__ __forceinline__ void apply_grad_update_pose_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                            FlexPose* out_x, const FlexPoseGradient* g,
                                                            const FlexTopo& flex_topo,
                                                            Real alpha){
    // -------------- torsion increment --------------
    for (int i_tor = 0; i_tor < flex_topo.ntorsion; i_tor++){
        apply_grad_update_dihe_warp(tile, out_x, &flex_topo, i_tor, g->dihedrals_g[i_tor] * alpha);
    }
    tile.sync();


    if (!FLAG_CONSTRAINT_DOCK){
        Real q[4] = {0}, tmp1[3] = {0}, tmp2[3] = {0};

        // -------------- orientation & center increment --------------
        // first, save the original center
        tmp1[0] = out_x->center[0];
        tmp1[1] = out_x->center[1];
        tmp1[2] = out_x->center[2];

        if (tile.thread_rank() == 0){
            // then update center
            out_x->center[0] = clamp_by_range(tmp1[0] + alpha * g->center_g[0], BOX_X_HI, BOX_X_LO);
            out_x->center[1] = clamp_by_range(tmp1[1] + alpha * g->center_g[1], BOX_Y_HI, BOX_Y_LO);
            out_x->center[2] = clamp_by_range(tmp1[2] + alpha * g->center_g[2], BOX_Z_HI, BOX_Z_LO);

            // and update orientation
            tmp2[0] = g->orientation_g[0] * alpha;
            tmp2[1] = g->orientation_g[1] * alpha;
            tmp2[2] = g->orientation_g[2] * alpha;
            rotvec_to_quaternion(q, tmp2);
            DPrint1("\nRotVec: %f, %f, %f, q: %f, %f, %f, %f\n", tmp2[0], tmp2[1], tmp2[2], q[0], q[1], q[2], q[3]);
            quaternion_increment(out_x->orientation, q);
            // DPrint1("q is %f, %f, %f, %f\n", q[0])

        }
        tile.sync();
        q[0] = tile.shfl(q[0], 0);
        q[1] = tile.shfl(q[1], 0);
        q[2] = tile.shfl(q[2], 0);
        q[3] = tile.shfl(q[3], 0);

        // Finally, add center_g to and rotate by orientation_g coords of all atoms
        for (int i_at = tile.thread_rank(); i_at < flex_topo.natom; i_at += tile.num_threads()){
            // locate rotation origin on the original center
            tmp2[0] = out_x->coords[i_at * 3] - tmp1[0];
            tmp2[1] = out_x->coords[i_at * 3 + 1] - tmp1[1];
            tmp2[2] = out_x->coords[i_at * 3 + 2] - tmp1[2];

            rotate_vec_by_quaternion(tmp2, q);

            // move to the new center
            out_x->coords[i_at * 3] = tmp2[0] + out_x->center[0];
            out_x->coords[i_at * 3 + 1] = tmp2[1] + out_x->center[1];
            out_x->coords[i_at * 3 + 2] = tmp2[2] + out_x->center[2];
        }
        tile.sync();
    }
}


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
        out_pose_new->orientation[0] = pose_old->orientation[0];
        out_pose_new->orientation[1] = pose_old->orientation[1];
        out_pose_new->orientation[2] = pose_old->orientation[2];
        out_pose_new->orientation[3] = pose_old->orientation[3];

        if (FLAG_CONSTRAINT_DOCK){
            // center and orientation are fixed
            aux_g->center_g[0] = 0;
            aux_g->center_g[1] = 0;
            aux_g->center_g[2] = 0;
            aux_g->orientation_g[0] = 0;
            aux_g->orientation_g[1] = 0;
            aux_g->orientation_g[2] = 0;
            aux_g->orientation_g[3] = 0;
        }
        else{
            // random center, set gradient
            tmp4[0] = get_real_within_by_int(ri4.x, BOX_X_LO, BOX_X_HI, ceil((BOX_X_HI - BOX_X_LO) / BOX_PREC) + 1);
            tmp4[1] = get_real_within_by_int(ri4.y, BOX_Y_LO, BOX_Y_HI, ceil((BOX_Y_HI - BOX_Y_LO) / BOX_PREC) + 1);
            tmp4[2] = get_real_within_by_int(ri4.z, BOX_Z_LO, BOX_Z_HI, ceil((BOX_Z_HI - BOX_Z_LO) / BOX_PREC) + 1);

            aux_g->center_g[0] = tmp4[0] - out_pose_new->center[0];
            aux_g->center_g[1] = tmp4[1] - out_pose_new->center[1];
            aux_g->center_g[2] = tmp4[2] - out_pose_new->center[2];

            // random orientation, set gradient
            rotvec_to_axis_angle(tmp4, rotvec); // tmp4 is axis_angle
            tmp4[0] = get_real_within_by_int(ri4.w, -PI, PI, ceil(2 * PI / TOR_PREC) + 1);
            aux_g->orientation_g[0] = tmp4[1] * tmp4[0];
            aux_g->orientation_g[1] = tmp4[2] * tmp4[0];
            aux_g->orientation_g[2] = tmp4[3] * tmp4[0]; //todo: change to a more efficient sampling
        }

        // generate random uints for all torsions
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


__device__ __forceinline__ void line_search_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                 const FixMol& fix_mol, const FixParamVina& fix_param,
                                                 const FlexParamVina& flex_param, const FlexPose* x,
                                                 const FlexPoseGradient* g, const FlexTopo& flex_topo,
                                                 const FlexPoseGradient* p,
                                                 Real* aux_f,
                                                 FlexPose* out_x_new, FlexPoseGradient* out_g_new,
                                                 const Real e0, int dim_g, int dim_x, Real* out_e, Real* out_alpha){
    Real alpha = 1.0; // step length, which is the target of BLS
    Real e_new = 0.; // energy of tried step

    const Real pg = g_dot_product_warp(tile, p, g, dim_g);

    int trial = 0;
    for (; trial < LINE_SEARCH_STEPS; trial++){
        atomicAdd(&funcCallCount, 1);  // todo: for debug, show call count

        duplicate_pose_warp(tile, out_x_new, x, dim_x, flex_topo.natom); // x_new = x
        // DPrint1("[[[after dup]]]  ---  dihe: %f, %f\n\n", out_x_new->dihedrals[0], out_x_new->dihedrals[1]);

        // apply alpha * gradient, get new x
        apply_grad_update_pose_warp(tile, out_x_new, p, flex_topo, alpha); // apply gradient increment
        // DPrint1("[[[after update]]]  ---  dihe: %f, %f\n\n", out_x_new->dihedrals[0], out_x_new->dihedrals[1]);

        e_new = cal_e_grad_warp(tile, out_x_new, out_g_new, flex_topo, fix_mol, flex_param, fix_param, aux_f);
        // DPrint1("\n[LINE SEARCH] coord0_new: %f, %f, %f coord10_new: %f, %f, %f e_new: %f, alpha: %f, p: %f, %f, %f, %f, %f, %f, pg: %f \n",
        //     out_x_new->coords[0], out_x_new->coords[1], out_x_new->coords[2],
        //     out_x_new->coords[30], out_x_new->coords[31], out_x_new->coords[32],
        //     e_new, alpha,
        //     p->center_g[0], p->center_g[1], p->center_g[2],
        //     p->orientation_g[0], p->orientation_g[1], p->orientation_g[2], pg);

        if (e_new - e0 < LINE_SEARCH_C0 * alpha * pg){
            // DPrint1("\nLine search SUCCEED!\n", 1);
            break;
        }
        //todo: Wolfe condition

        alpha *= LINE_SEARCH_MULTIPLIER; // lower step length
    }
    tile.sync();

    // fixme: WHY? this version will lead to active lock (100% occupancy of GPU, never stop)
    // fixme: requiring fact-checking @25-03-20
    // if (tile.thread_rank() == 0){ // check why the active lock appears
    //     *out_e = e_new;
    //     *out_alpha = alpha;
    // }


    // this version works, no lock
    *out_e = e_new; //
    *out_alpha = alpha;

    tile.sync();
}


/**
 * @brief Update Hessian matrix for BFGS
 * @param out_h Hessian matrix, triangular matrix
 * @param dim Dimension of Hessian & gradient
 * @param p vector p
 * @param y vector y
 * @param aux_minus_hy Auxiliary vector -Hy
 * @param alpha Step length
 */
__forceinline__ __device__ void bfgs_update_hessian_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                         Real* out_h, int dim, const FlexPoseGradient* p,
                                                         const FlexPoseGradient* y,
                                                         FlexPoseGradient* aux_minus_hy, const Real alpha){
    Real yp = 0, yhy = 0;
    yp = g_dot_product_warp(tile, y, p, dim);
    if (alpha * yp < EPSILON_cu){
        return;
    }

    // minus_hy = -hy (dim*1 vector)
    minus_mat_vec_product_warp(tile, aux_minus_hy, out_h, y, dim);

    // y^THy
    yhy = -g_dot_product_warp(tile, y, aux_minus_hy, dim);
    Real r = 1 / (alpha * yp); //1 / (s^T * y) , where s = alpha * p

    Real tmp = 0;
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()){
        for (int j = i; j < dim; j++){
            tmp = alpha * r * (grad_index_read(aux_minus_hy, i) * grad_index_read(p, j) +
                    grad_index_read(aux_minus_hy, j) * grad_index_read(p, i))
                + alpha * alpha * (r * r * yhy + r) * grad_index_read(p, i) * grad_index_read(p, j);
            // s * s == alpha * alpha * p * p

            out_h[i + j * (j + 1) / 2] += tmp;
        }
    }
    tile.sync();
}

SCOPE_INLINE void print_uptri_mat(Real* mat, int dim){
    for (int i = 0; i < dim; i++){
        for (int j = 0; j < dim; j++){
            if (j < i){
                printf("      ");
            }
            else{
                printf("%.3f ", mat[uptri_mat_index(i, j)]);
            }
        }
        printf("\n");
    }
}

SCOPE_INLINE void print_g(const FlexPoseGradient* g, int dim){
    DPrint("\nG: Center: ", 1);
    for (int i = 0; i < 3; i ++){
        DPrint("%f ", g->center_g[i]);
    }
    DPrint("Orientation: ", 1);
    for (int i = 3; i < 7; i ++){
        DPrint("%f ", g->orientation_g[i - 3]);
    }
    DPrint("Dihedrals: ", 1);
    for (int i = 7; i < dim; i ++){
        DPrint("%f ", g->dihedrals_g[i - 7]);
    }
    DPrint("\n", 1);

}


__forceinline__ __device__ void bfgs_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                          FlexPose* out_x, const FlexTopo& flex_topo, const FixMol& fix_mol,
                                          const FlexParamVina& flex_param, const FixParamVina& fix_param,
                                          FlexPose* aux_x_new, FlexPose* aux_x_ori,
                                          FlexPoseGradient* aux_g, FlexPoseGradient* aux_g_new,
                                          FlexPoseGradient* aux_g_ori,
                                          FlexPoseGradient* aux_p, FlexPoseGradient* aux_y,
                                          FlexPoseGradient* aux_minus_hy,
                                          FlexPoseHessian* aux_h, FlexForce* aux_f, int max_steps){
    int dim_g = dof_g + flex_topo.ntorsion;
    int dim_x = dof_x + flex_topo.ntorsion; // center, orientation, torsion

    Real E_ori = cal_e_grad_warp(tile, out_x, aux_g, flex_topo, fix_mol, flex_param, fix_param, aux_f->f);

    // Record initial energy and gradient. E_ori is energy, aux_g is set as current gradient
    // alpha is step length
    Real E = E_ori, E1 = 0, alpha = 0;
    DPrint1("\nBefore BFGS, Energy is %f\n", E_ori);
    // print_g(aux_g, dim);


    // Initialize hessian as Unit Matrix
    init_tri_mat_warp(tile, aux_h->matrix, dim_g, 0); // set zero
    set_tri_mat_diagonal_warp(tile, aux_h->matrix, dim_g, 1); // set diagonal to 1

    // Initialize aux_g_new
    duplicate_grad_warp(tile, aux_g_new, aux_g, dim_g);
    // Initialize aux_pose_new
    duplicate_pose_warp(tile, aux_x_new, out_x, dim_x, flex_topo.natom);

    // Record gradient before optimization: aux_g_ori = aux_g
    duplicate_grad_warp(tile, aux_g_ori, aux_g, dim_g);
    // aux_pose_ori := copy of out_pose
    duplicate_pose_warp(tile, aux_x_ori, out_x, dim_x, flex_topo.natom);

    // aux_p := copy of aux_g, just for initialization
    duplicate_grad_warp(tile, aux_p, aux_g, dim_g);

    for (int step = 0; step < max_steps; step++){
        // compute line search direction aux_p = -Hg (dim*1 vector)
        minus_mat_vec_product_warp(tile, aux_p, aux_h->matrix, aux_g, dim_g);

        // find the best alpha, and updates x_new & aux_g_new. f1 is the new energy
        line_search_warp(tile, fix_mol, fix_param, flex_param, out_x, aux_g, flex_topo,
                         aux_p, aux_f->f, aux_x_new, aux_g_new, E, dim_g, dim_x, &E1, &alpha);
        DPrint1("\nAlpha is %f, New Energy is %f\n", alpha, E1);

        // aux_y = aux_g_new
        duplicate_grad_warp(tile, aux_y, aux_g_new, dim_g);

        // aux_y = aux_y - aux_g, namely aux_y = aux_g_new - aux_g
        for (int i = tile.thread_rank(); i < dim_g; i += tile.num_threads()){
            Real tmp = grad_index_read(aux_y, i) - grad_index_read(aux_g, i);
            grad_index_write(aux_y, i, tmp);
        }
        tile.sync();

        // Update energy as the new one
        E = E1;

        // out_x := aux_x_new
        duplicate_pose_warp(tile, out_x, aux_x_new, dim_x, flex_topo.natom);

        // Stop criterion: check convergence todo: why not checking g_new?
        Real gg = g_dot_product_warp(tile, aux_g, aux_g, dim_g);
        if (sqrtf(gg) < 1e-5f){
            break;
        }

        // Update aux_g:= aux_g_new
        duplicate_grad_warp(tile, aux_g, aux_g_new, dim_g);

        // Choose a better initial hessian
        if (step == 0){
            Real yy = g_dot_product_warp(tile, aux_y, aux_y, dim_g);
            if (fabs(yy) > EPSILON_cu){
                // yp = aux_y * -Hg
                Real yp = g_dot_product_warp(tile, aux_y, aux_p, dim_g);
                set_tri_mat_diagonal_warp(tile, aux_h->matrix, dim_g, alpha * yp / yy); // heuristic value
            }
        }
        tile.sync();

        // aux_minus_hy serves a container rather than a given parameter
        bfgs_update_hessian_warp(tile, aux_h->matrix, dim_g, aux_p, aux_y, aux_minus_hy, alpha);
    }

    // If this optimization fails (a higher energy is found), resume to the original state
    if (E > E_ori){
        E = E_ori;
        duplicate_pose_warp(tile, out_x, aux_x_ori, dim_x, flex_topo.natom);
        duplicate_grad_warp(tile, aux_g, aux_g_ori, dim_g);
    }

    // write output_type_cuda energy
    if (tile.thread_rank() == 0){
        out_x->energy = E;
    }
    tile.sync();
}


#endif //BFGS_CUH
