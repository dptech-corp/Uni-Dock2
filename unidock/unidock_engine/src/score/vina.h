//
// Created by lccdp on 24-8-15.
//

#ifndef VINA_H
#define VINA_H

#include "model/model.h"
#include "myutils/common.h"
#include "myutils/mymath.h"
#include "constants/constants.h"


/**
 * @brief Vina-required parameters and additional info about flex mol
 */
struct FlexParamVina{
    // One copy for one ligand, poses share this
    int npair_intra; // flex-flex
    int* pairs_intra; // size: npair_intra * 2. each two is a pair. The bound is loose, so never changed after initialization.
    Real* r1_plus_r2_intra; // size: npair_intra. vdW radii summations of each pair, intra part + inter part

    int npair_inter; // flex-protein
    int* pairs_inter; // size: npair_inter * 2. each two is a pair: (index_flex, index_protein)
    Real* r1_plus_r2_inter; // size: npair_inter. vdW radii summations of each pair, intra part + inter part

    int* atom_types; // size: natom
};
FlexParamVina* alloccp_FlexParamVina_gpu(const FlexParamVina& flex_param_vina, int natom);
void free_FlexParamVina_gpu(FlexParamVina* flex_param_vina_cu);


/**
 * @brief Vina-required parameters and additional info about fixed mol
 */
struct FixParamVina{
    int* atom_types; // size: natom
};
FixParamVina* alloccp_FixParamVina_gpu(const FixParamVina& fix_param_vina, int natom);
void free_FixParamVina_gpu(FixParamVina* fix_param_vina);



struct VinaScore{
    // todo: further modification
    Real e = 0;
    Real e_inter = 0;
    Real e_intra = 0;
};


SCOPE_INLINE bool vn_is_hydrophobic(int t){
    return t == VN_TYPE_C_H || t == VN_TYPE_F_H || t == VN_TYPE_Cl_H || t == VN_TYPE_Br_H
        || t == VN_TYPE_I_H;
}

SCOPE_INLINE bool vn_is_acceptor(int t){
    return t == VN_TYPE_N_A || t == VN_TYPE_N_DA || t == VN_TYPE_O_A || t == VN_TYPE_O_DA;
}

SCOPE_INLINE bool xs_is_donor(int t){
    return t == VN_TYPE_N_D || t == VN_TYPE_N_DA || t == VN_TYPE_O_D || t == VN_TYPE_O_DA
        || t == VN_TYPE_Met_D;
}

SCOPE_INLINE bool xs_h_bond_possible(int t1, int t2){
    return (vn_is_acceptor(t1) && xs_is_donor(t2)) || (xs_is_donor(t1) && vn_is_acceptor(t2));
}

/**
 * @brief Linear interpolation between (x_lo, 1) and (x_hi, 0)
 * @param x_lo Lower bound
 * @param x_hi Upper bound
 * @param x Input value
 * @param out_f Output derivative
 * @return Interpolated value
 */
SCOPE_INLINE Real slope_interp(Real x_lo, Real x_hi, Real x, Real* out_f){
    // assert(x_lo < x_hi)
    if (x <= x_lo){
        *out_f = 0.;
        return 1;
    }
    if (x >= x_hi){
        *out_f = 0.;
        return 0;
    }
    *out_f = 1 / (x_lo - x_hi);
    return (x_hi - x) / (x_hi - x_lo);
}


/**
 * @brief Define a class to save one copy of parameters and functions.
 * Each model should be put to Vina for calculations?
 */
class Vina{
private:
    // Use one cutoff

    const Real gauss1_offset = 0;
    const Real gauss1_width = .5;
    // const Real gauss1_cutoff = 8.0;
    const Real weight_gauss1 = -0.035579;

    const Real gauss2_offset = 3;
    const Real gauss2_width = 2.0;
    // const Real gauss2_cutoff = 8.0;
    const Real weight_gauss2 = -0.005156;

    const Real repulsion_offset = 0.0;
    const Real repulsion_width = 8.0;
    // const Real repulsion_cutoff = 8.0;
    const Real weight_repulsion = 0.840245;

    const Real hydrophobic_low = 0.5;
    const Real hydrophobic_high = 1.5;
    // const Real hydrophobic_cutoff = 8.0;
    const Real weight_hydrophobic = -0.035069;

    const Real hbond_low = -0.7;
    const Real hbond_high = 0;
    // const Real hbond_cutoff = 8.0;
    const Real weight_hbond = -0.587439;

    const Real weight_conf_indep = 0.05846;

    void cal_pair_dists(Real* __restrict__ coords_fix, Real* __restrict__ coords_flex, int* pair_list, int npair,
                        Real* out_dists);


public:
    const Real r2_cutoff = 64.0;

    /**
     * @brief Compute the value and derivative of the first Gaussian function
     * @snippet test/unit/score/test_vina.cpp vina_gaussian1
     * @param d surface distance
     * @param out_f Output derivative
     * @return energy
     */
    SCOPE_INLINE Real vina_gaussian1(Real d, Real* out_f){
        return gaussian(d, gauss1_offset, gauss1_width, out_f);
    }

    /**
     * @brief Compute the value and derivative of the second Gaussian function
     * @snippet test/unit/score/test_vina.cpp vina_gaussian2
     * @param d surface distance
     * @param out_f Output derivative
     * @return energy
     */
    SCOPE_INLINE Real vina_gaussian2(Real d, Real* out_f){
        return gaussian(d, gauss2_offset, gauss2_width, out_f);
    }

    /**
     * @brief Compute the value and derivative of the repulsion function
     * @snippet test/unit/score/test_vina.cpp vina_repulsion
     * @param d surface distance
     * @param out_f Output derivative
     * @return energy
     */
    SCOPE_INLINE Real vina_repulsion(Real d, Real* out_f){
        if (d >= 0.0){
            *out_f = 0.0;
            return 0.0;
        }
        *out_f = 2 * d;
        return d * d;
    }


    /**
     * @brief Compute the value and derivative of the hydrophobic function
     * @snippet test/unit/score/test_vina.cpp vina_hydrophobic
     * @param d surface distance
     * @param out_f Output derivative
     * @return energy
     */
    SCOPE_INLINE Real vina_hydrophobic(Real d, Real* out_f){
        return slope_interp(hydrophobic_low, hydrophobic_high, d, out_f);
    }


    /**
     * @brief Compute the value and derivative of the hydrogen bond function
     * @snippet test/unit/score/test_vina.cpp vina_hbond
     * @param d surface distance
     * @param out_f Output derivative
     * @return energy
     */
    SCOPE_INLINE Real vina_hbond(Real d, Real* out_f){
        return slope_interp(hbond_low, hbond_high, d, out_f);
    }


    /**
     * @brief Compute the value of the conformation-independent term
     * @snippet test/unit/score/test_vina.cpp vina_conf_indep
     * @param e Inter-molecular energy between flex and protein
     * @param num_rotator Number of rotatable dihedrals
     * @return energy
     */
    SCOPE_INLINE Real vina_conf_indep(Real e, int num_rotator){
        return e / (1 + weight_conf_indep * num_rotator);
    }


    /**
     * @brief Compute the value and derivative of the geometrical energy. Positive d means to-origin
     * @snippet test/unit/score/test_vina.cpp eval_ef
     * @param d surface distance
     * @param at1 Atom type of the first atom
     * @param at2 Atom type of the second atom
     * @param out_f Output derivative (-force)
     * @return energy
     */
    SCOPE_INLINE Real eval_ef(Real d, int at1, int at2, Real* out_f){
        // add all items of geometrical energy
        Real e = 0;
        *out_f = 0;
        Real f = 0;

        // ignore all hydrogen atoms
        if (at1 == VN_TYPE_H || at2 == VN_TYPE_H){
            return e;
        }

        e += weight_gauss1 * vina_gaussian1(d, &f);
        *out_f += weight_gauss1 * f;

        e += weight_gauss2 * vina_gaussian2(d, &f);
        *out_f += weight_gauss2 * f;

        e += weight_repulsion * vina_repulsion(d, &f);
        *out_f += weight_repulsion * f;

        if (vn_is_hydrophobic(at1) && vn_is_hydrophobic(at2)){
            e += weight_hydrophobic * vina_hydrophobic(d, &f);
            *out_f += weight_hydrophobic * f;
        }
        if (xs_h_bond_possible(at1, at2)){
            e += weight_hbond * vina_hbond(d, &f);
            *out_f += weight_hbond * f;
        }
        return e;
    }



};


#endif //VINA_H
