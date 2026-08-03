//
// Created by Congcong Liu on 25-8-21.
//

#ifndef BIAS_H
#define BIAS_H
#include "vina.h"


class Bias{
    // achieve different biases here

    SCOPE_INLINE Real eval_ef(Real* r_, Real p1, Real p2, Real* out_f){
        //todo: allow user-flag to decide using which bias
        return eval_ef_pos(r_, p1, p2, out_f);
    }

    SCOPE_INLINE Real eval_ef_pos(Real* r_, Real V_set, Real r2, Real* out_f){
        // r_ is bias_position - flex_atom_position

        Real rr = r_[0] * r_[0] + r_[1] * r_[1] + r_[2] * r_[2];
        Real e_bias = V_set * expf(- rr / r2);

        out_f[0] += e_bias * 2 * r_[0] / r2;
        out_f[1] += e_bias * 2 * r_[1] / r2;
        out_f[2] += e_bias * 2 * r_[2] / r2;

        return e_bias;
    }

    SCOPE_INLINE Real eval_ef_zalign(Real* r_, Real a_qt, int vn_type, Real* out_f){

        // attractive

        const Real s_a = 30.;
        const Real c_a = 1.;
        Real k_a = vn_type == VN_TYPE_H ? 0.1 : 30.;
        Real r = cal_norm(r_);

        Real e_item = exp(s_a * (c_a - r));
        Real e_bias = a_qt * k_a / (1 + e_item);
        Real f = e_bias * e_item * (1 + e_item) * s_a;

        out_f[0] += f * r_[0] / r;
        out_f[1] += f * r_[0] / r;
        out_f[2] += f * r_[0] / r;

        return e_bias;
    }
};








#endif //BIAS_H
