//
// Created by Congcong Liu on 24-12-26.
//

#include "score.h"
#include "vina.h"


Vina SF;


void score(FlexPose* out_pose, const Real* flex_coords, const UDFixMol& udfix_mol, const UDFlexMol& udflex_mol,
           const DockParam& dock_param){
    Real rr = 0;
    Real f = 0;
    Real e_intra = 0., e_inter = 0., e_penalty = 0.;


    // 1. Compute Pairwise energy and forces
    // -- Compute intra-molecular energy (per-atom adjacency list)
    for (int i1 = 0; i1 < udflex_mol.natom; i1++){
        int start = udflex_mol.intra_range[i1 * 2];
        int count = udflex_mol.intra_range[i1 * 2 + 1];
        Real vdw_r1 = VN_VDW_RADII[udflex_mol.vina_types[i1]];

        for (int k = start; k < start + count; k++){
            int i2 = udflex_mol.intra_pairs[k];
            if (i2 <= i1){
                continue;
            }

            Real r_vec[3] = {
                flex_coords[i2 * 3] - flex_coords[i1 * 3],
                flex_coords[i2 * 3 + 1] - flex_coords[i1 * 3 + 1],
                flex_coords[i2 * 3 + 2] - flex_coords[i1 * 3 + 2]
            };
            rr = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];

            if (rr < SF.r2_cutoff){
                rr = sqrt(rr);
                e_intra += SF.eval_ef(rr - (vdw_r1 + VN_VDW_RADII[udflex_mol.vina_types[i2]]),
                                      udflex_mol.vina_types[i1], udflex_mol.vina_types[i2], &f);
            }
        }
    }

    out_pose->center[0] = e_intra;

    // -- Compute inter-molecular energy: flex-protein (double loop, no pair list)
    for (int i1 = 0; i1 < udflex_mol.natom; i1++){
        if (udflex_mol.vina_types[i1] == VN_TYPE_H) continue;
        Real vdw_r1 = VN_VDW_RADII[udflex_mol.vina_types[i1]];

        for (int i2 = 0; i2 < udfix_mol.natom; i2++){
            if (udfix_mol.vina_types[i2] == VN_TYPE_H) continue;

            Real r_vec[3] = {
                udfix_mol.coords[i2 * 3] - flex_coords[i1 * 3],
                udfix_mol.coords[i2 * 3 + 1] - flex_coords[i1 * 3 + 1],
                udfix_mol.coords[i2 * 3 + 2] - flex_coords[i1 * 3 + 2]
            };
            rr = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];

            if (rr < SF.r2_cutoff){
                rr = sqrt(rr);
                Real e = SF.eval_ef(rr - (vdw_r1 + VN_VDW_RADII[udfix_mol.vina_types[i2]]),
                                    udflex_mol.vina_types[i1], udfix_mol.vina_types[i2], &f);
                e_inter += e;
            }
        }
    }

    out_pose->center[1] = e_inter;

    Real tmp3[3];
    // -- Compute inter-molecular energy: penalty
    for (int i = 0; i < udflex_mol.natom; i++){
        if (udflex_mol.vina_types[i] == VN_TYPE_H){
            continue;
        }
        e_penalty += cal_box_penalty(flex_coords + i * 3, dock_param.box, tmp3);;
    }
    out_pose->center[2] = e_penalty;

    // 1.4. Compute position-bias
    Real e_bias = 0.;
    for (auto & b: udflex_mol.biases){
        Real f_bias[3] = {0.};
        Real coord_adj[3] = {
            flex_coords[b.i * 3],
            flex_coords[b.i * 3 + 1],
            flex_coords[b.i * 3 + 2]
        };

        Real r_[3] = {
            b.param[0] -  coord_adj[0],
            b.param[1] -  coord_adj[1],
            b.param[2] -  coord_adj[2]
        };

        if (dock_param.bias_type == BT_POS){
            e_bias += SF.eval_ef_pos(r_, b.param[3] * dock_param.bias_k, b.param[4], f_bias);
        } else if (dock_param.bias_type == BT_ALIGN){
            e_bias += SF.eval_ef_zalign(r_, b.param[3] * dock_param.bias_k, udflex_mol.vina_types[b.i], f_bias);
        }
    }
    out_pose->rot_vec[3] = e_bias;
}


/**
 * @brief Decompose inter-molecular energy per flex atom into 5 Vina terms.
 *        CPU-only, called after the final scoring stage.
 */
void score_decomp(std::vector<AtomEnergyDecomp>& out_decomp,
                  const Real* flex_coords, const UDFixMol& udfix_mol,
                  const UDFlexMol& udflex_mol){
    out_decomp.assign(udflex_mol.natom, AtomEnergyDecomp{});

    for (int i1 = 0; i1 < udflex_mol.natom; i1++){
        if (udflex_mol.vina_types[i1] == VN_TYPE_H) continue;
        Real vdw_r1 = VN_VDW_RADII[udflex_mol.vina_types[i1]];

        for (int i2 = 0; i2 < udfix_mol.natom; i2++){
            if (udfix_mol.vina_types[i2] == VN_TYPE_H) continue;

            Real r_vec[3] = {
                udfix_mol.coords[i2 * 3] - flex_coords[i1 * 3],
                udfix_mol.coords[i2 * 3 + 1] - flex_coords[i1 * 3 + 1],
                udfix_mol.coords[i2 * 3 + 2] - flex_coords[i1 * 3 + 2]
            };
            Real rr = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];

            if (rr < SF.r2_cutoff){
                rr = sqrt(rr);
                Real terms[DECOMP_N_TERMS];
                SF.eval_decomp(rr - (vdw_r1 + VN_VDW_RADII[udfix_mol.vina_types[i2]]),
                               udflex_mol.vina_types[i1], udfix_mol.vina_types[i2], terms);
                out_decomp[i1].gauss1 += terms[0];
                out_decomp[i1].gauss2 += terms[1];
                out_decomp[i1].repulsion += terms[2];
                out_decomp[i1].hydrophobic += terms[3];
                out_decomp[i1].hbond += terms[4];
            }
        }
    }
}
