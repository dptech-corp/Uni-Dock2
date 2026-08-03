//
// Created by lccdp on 24-8-15.
//

#ifndef FF_GAFF2_H
#define FF_GAFF2_H

#include "model/model.h"


/**
 * @brief Vina-required parameters and additional info about flex mol
 */
struct FlexParamGaff2{
    int* nonbond_list; // starts with [0; 1, 8; 3, 8, 99, 18]
    int* atom_types;
};

struct FixParamGaff2{
    int* atom_types; // search for charge, mass
};

// loop on one thread? no, keeps at least one warp for each pose to parallel
struct Gaff2Data{
    Real* nonbond_dists; // distance between each pair of atoms
};


struct Gaff2Score{ // todo: further modification
    Real e = 0;
    Real e_inter = 0;
    Real e_intra = 0;
};

/**
 * @brief Define a class to save one copy of parameters and functions.
 * Each model should be put to Vina for calculations?
 */
class Gaff2{
    const Real gauss1_offset = 0;
    const Real gauss1_width = .5;
    const Real gauss1_cutoff = 8.0;
    const Real weight_gauss1 = -0.035579;

    const Real gauss2_offset = 3;
    const Real gauss2_width = 2.0;
    const Real gauss2_cutoff = 8.0;
    const Real weight_gauss2 = -0.005156;

    const Real repulsion_offset = 0.0;
    const Real repulsion_width = 8.0;
    const Real repulsion_cutoff = 8.0;
    const Real weight_repulsion = 0.840245;

    const Real hydrophobic_low = 0.5;
    const Real hydrophobic_high = 1.5;
    const Real hydrophobic_cutoff = 8.0;
    const Real weight_hydrophobic = -0.035069;

    const Real hbond_low = -0.7;
    const Real hbond_high = 0;
    const Real hbond_cutoff = 8.0;
    const Real weight_hydrogen = -0.587439;

    const Real conf_independ = 0;
    const Real weight_conf_indep = 0.05846;

    void cal_pair_dists(Real* __restrict__ coords_fix, Real* __restrict__ coords_flex, int * pair_list, int npair,
        Real* out_dists);

    Gaff2Score update_dist_and_score(const FixMol& fix_mol,
        const FlexPose& flex_pose, const FlexParamGaff2 & flex_param,
        Gaff2Data* out_vina_data){

        // compute distances

        // compute vina score
        Gaff2Score score;

        return score;
    }

    Real gaff2_diheral(Real r);
    Real gaff2_electrostatic(Real r);
    Real gaff2_vdW(Real r);
};



#endif //FF_GAFF2_H
