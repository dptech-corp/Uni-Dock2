//
// Created by Congcong Liu on 24-11-21.
//

#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include "model/model.h"
#include "score/vina.h"

#include "constants/constants.h"

void optimize_cu(FlexPose* out_poses, const int* pose_inds, const FlexTopo* flex_topos, const FixMol& fix_mol,
                          const FlexParamVina* flex_params, const FixParamVina& fix_param,
                          FlexPose* aux_poses, FlexPoseGradient* aux_gradients, FlexPoseHessian* aux_hessians,
                          FlexForce* aux_forces,
                          int refine_steps, int nblock, int npose_per_flex,
                          Real v_cap = V_CAP_AUTHENTIC, Real slope = PENALTY_SLOPE);


#endif //OPTIMIZE_H
