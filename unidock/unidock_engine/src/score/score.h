//
// Created by Congcong Liu on 24-12-26.
//

#ifndef SCORE_H
#define SCORE_H
#include "myutils/common.h"
#include "constants/constants.h"
#include "model/model.h"
#include <vector>

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


void score(FlexPose* out_pose, const Real* flex_coords, const UDFixMol& udfix_mol, const UDFlexMol& udflex_mol,
           const DockParam& dock_param);



#define DECOMP_N_TERMS 5
struct AtomEnergyDecomp {
    Real gauss1 = 0;
    Real gauss2 = 0;
    Real repulsion = 0;
    Real hydrophobic = 0;
    Real hbond = 0;
};

void score_decomp(std::vector<AtomEnergyDecomp>& out_decomp,
                  const Real* flex_coords, const UDFixMol& udfix_mol,
                  const UDFlexMol& udflex_mol);


#endif //SCORE_H
