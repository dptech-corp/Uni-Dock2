//
// Created by Congcong Liu on 24-12-26.
//

#ifndef SCORE_H
#define SCORE_H
#include "myutils/common.h"


SCOPE_INLINE Real cal_box_penalty(const Real* coord, const Box& box, Real* out_f){
    Real penalty = 0.;

    penalty += (coord[0] < box.x_lo) * (box.x_lo - coord[0]);
    out_f[0] += (coord[0] < box.x_lo) * (-PENALTY_SLOPE);

    penalty += (coord[0] > box.x_hi) * (coord[0] - box.x_hi);
    out_f[0] += (coord[0] > box.x_hi) * (PENALTY_SLOPE);

    penalty += (coord[1] < box.y_lo) * (box.y_lo - coord[1]);
    out_f[1] += (coord[1] < box.y_lo) * (-PENALTY_SLOPE);

    penalty += (coord[1] > box.y_hi) * (coord[1] - box.y_hi);
    out_f[1] += (coord[1] > box.y_hi) * (PENALTY_SLOPE);

    penalty += (coord[2] < box.z_lo) * (box.z_lo - coord[2]);
    out_f[2] += (coord[2] < box.z_lo) * (-PENALTY_SLOPE);

    penalty += (coord[2] > box.z_hi) * (coord[2] - box.z_hi);
    out_f[2] += (coord[2] > box.z_hi) * (PENALTY_SLOPE);

    return penalty * PENALTY_SLOPE;
}


Vina SF;


void score(FlexPose* out_pose, const Real* flex_coords, const UDFixMol& udfix_mol, const UDFlexMol& udflex_mol,
           const Box& box){
    Real rr = 0;
    Real f = 0;
    Real e_intra = 0., e_inter = 0., e_penalty = 0.;


    // 1. Compute Pairwise energy and forces
    // -- Compute intra-molecular energy
    for (int i = 0; i < udflex_mol.intra_pairs.size() / 2; i++){
        int i1 = udflex_mol.intra_pairs[i * 2], i2 = udflex_mol.intra_pairs[i * 2 + 1];

        // Cartesian distances won't be saved
        Real r_vec[3] = {
            // cuda vector multiply v3 v4
            flex_coords[i2 * 3] - flex_coords[i1 * 3],
            flex_coords[i2 * 3 + 1] - flex_coords[i1 * 3 + 1],
            flex_coords[i2 * 3 + 2] - flex_coords[i1 * 3 + 2]
        };
        rr = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];
        // DPrintCPU("Pair: %d %d r2 = %f", i1, i2, rr);
        if (rr < SF.r2_cutoff){
            rr = sqrt(rr); // use r2 as a container for |r|
            // if ((i1 == 17) and i2 == 20){
            //     int mmm = 1;
            //     for (int n = 200; n < 500; n++) {
            //         float r = n * 0.01;
            //         float tmp_f=0;
            //         float res = SF.eval_ef(r - udflex_mol.r1_plus_r2_intra[i], udflex_mol.vina_types[i1],
            //                         udflex_mol.vina_types[i2], &tmp_f);
            //         // printf("r=%f, e=%f\n", r, res);
            //     }
            // }
            Real tmp = SF.eval_ef(rr - udflex_mol.r1_plus_r2_intra[i], udflex_mol.vina_types[i1],
                                  udflex_mol.vina_types[i2], &f);
            e_intra += tmp;
            // DPrintCPU(" e = %f", tmp);
        }
        // DPrintCPU("\n", 1);
    }
    // DPrintCPU("\n", 1);

    out_pose->center[0] = e_intra;

    // -- Compute inter-molecular energy: flex-protein
    int atom_id = udflex_mol.inter_pairs[0]; // begin from
    Real atom_e = 0;
    for (int i = 0; i < udflex_mol.inter_pairs.size() / 2; i++){
        int i1 = udflex_mol.inter_pairs[i * 2], i2 = udflex_mol.inter_pairs[i * 2 + 1];
        if (i1 > atom_id){
            // DPrintCPU("Atom %d e_inter = %f\n", atom_id, atom_e);
            atom_e = 0;
            atom_id++;
        }
        // Cartesian distances won't be saved
        Real r_vec[3] = {
            udfix_mol.coords[i2 * 3] - flex_coords[i1 * 3],
            udfix_mol.coords[i2 * 3 + 1] - flex_coords[i1 * 3 + 1],
            udfix_mol.coords[i2 * 3 + 2] - flex_coords[i1 * 3 + 2]
        };
        rr = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];

        if (rr < SF.r2_cutoff){
            rr = sqrt(rr); // use r2 as a container for |r|
            Real e = SF.eval_ef(rr - udflex_mol.r1_plus_r2_inter[i], udflex_mol.vina_types[i1],
                                udfix_mol.vina_types[i2], &f);
            // printf("Atom %d %d e=%f\n", i1, i2, e);
            e_inter += e;
            atom_e += e;
        }
    }
    // DPrintCPU("Atom %d e_inter = %f\n", atom_id, atom_e);

    out_pose->center[1] = e_inter;

    Real tmp3[3];
    // -- Compute inter-molecular energy: penalty
    for (int i = 0; i < udflex_mol.natom; i++){
        if (udflex_mol.vina_types[i] == VN_TYPE_H){
            continue;
        }
        e_penalty += cal_box_penalty(flex_coords + i * 3, box, tmp3);;
    }
    out_pose->center[2] = e_penalty;
}


#endif //SCORE_H
