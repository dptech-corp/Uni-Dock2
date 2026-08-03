//
// Created by lccdp on 24-8-15.
//

#ifndef UD2_MC_H
#define UD2_MC_H

#include "model/model.h"
#include "score/vina.h"
#include <curand_kernel.h>

SCOPE_INLINE bool metropolis_accept(Real e_old, Real e_new, Real temperature, Real rand_x) {
    if (e_new < e_old){
        return true;
    }
    Real accept_p = exp((e_old - e_new) / temperature);
    return rand_x < accept_p;
}


/**
 * @brief Use Monte Carlo to search for the best pose, starting from out_pose.
 */
void mc_cu(FlexPose* out_poses, const FlexTopo* topos,
           const FixMol& fix_mol, const FlexParamVina* flex_param, const FixParamVina& fix_param,
           FlexPose* aux_poses, FlexPoseGradient* aux_gradients, FlexPoseHessian* aux_hessians, FlexForce* aux_forces,
           int mc_steps, int opt_steps, int nflex, int exhuastiveness, int seed, bool randomize);




#endif //UD2_MC_H
