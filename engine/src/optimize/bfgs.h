//
// Created by Congcong Liu on 24-12-9.
//

#ifndef BFGS_H
#define BFGS_H

#include "model/model.h"
#include "score/vina.h"

void line_search_one_gpu(const FixMol& fix_mol, const FixParamVina& fix_param,
                     const FlexParamVina& flex_param, const FlexPose& x,
                     const FlexTopo& flex_topo,
                     FlexPose* out_x_new, FlexPoseGradient* out_g_new,
                     Real* out_e, Real* out_alpha);


void cal_e_grad_one_gpu(const FixMol& fix_mol, const FixParamVina& fix_param,
        const FlexParamVina& flex_param, const FlexPose& x, const FlexTopo& flex_topo,
        FlexPoseGradient* out_g, Real* out_e);


#endif //BFGS_H
