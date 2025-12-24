//
// Created by lccdp on 24-8-15.
// Models defined here are logical-physical models, rather than computation-efficiency models.
// They can be obtained from input files.
//

#ifndef MOLECULE_H
#define MOLECULE_H

#include <vector>
#include <string>
#include <set>
#include <spdlog/spdlog.h>

#include "myutils/common.h"
#include "myutils/errors.h"
// ==================  For a single flexible molecule ==================
struct FlexPose{
    Real energy = 999;
    Real center[3]={0.};
    Real rot_vec[4]={0.}; // # record the rot_vec of last rotation todo: move to FlexPoseGradient
    Real* coords; // size: natom * 3
    Real* dihedrals; // size: ntorsion. Unit: radian (not degree)
};
FlexPose* alloccp_FlexPose_gpu(const FlexPose& flex_pose, int natom, int ntorsion);
void free_FlexPose_gpu(FlexPose* flex_pose_cu);
void freecp_FlexPose_gpu(FlexPose* flex_pose_cu, FlexPose* out_flex_pose, int natom, int ntorsion);


/**
 * @brief Gradient.
 */
struct FlexPoseGradient{
    Real center_g[3] = {0}; // translation gradient (dx, dy, dz)
    Real orientation_g[4] = {0.}; // Only the top 3 are used (-torque) todo: size from 4 to 3
    Real* dihedrals_g; // size: ntorsion. dihedrals_g
};
FlexPoseGradient* alloccp_FlexPoseGradient_gpu(const FlexPoseGradient& flex_pose_gradient, int ntorsion);
void free_FlexPoseGradient_gpu(FlexPoseGradient* flex_pose_gradient);
void freecp_FlexPoseGradient_gpu(FlexPoseGradient* flex_pose_gradient_cu, FlexPoseGradient* out_flex_pose_gradient, int ntorsion);

# define dof_g  6 // DOF of grad: 3 + 3(-torque)
# define dof_x  7 // DOF of pose: 3 + 4(quaternion)

SCOPE_INLINE void grad_index_write(FlexPoseGradient* out_g, int index, Real value){
    if (index < 3){
        out_g->center_g[index] = value;
    }
    else if (index < dof_g){
        out_g->orientation_g[index - 3] = value;
    }
    else{
        out_g->dihedrals_g[index - dof_g] = value;
    }
}


SCOPE_INLINE Real grad_index_read(const FlexPoseGradient* g, int index){
    if (index < 3){
        return g->center_g[index];
    }
    if (index < dof_g){
        return g->orientation_g[index - 3];
    }

    return g->dihedrals_g[index - dof_g];
}


/**
 * @brief Hessian matrix
 */
struct FlexPoseHessian{
    Real* matrix; // size: dim * (dim + 1) / 2. dim = 3(center) + 4(orientation) + ntorsion
};
FlexPoseHessian* alloc_FlexPoseHessian_gpu(int ntorsion);
FlexPoseHessian* alloccp_FlexPoseHessian_gpu(const FlexPoseHessian& flex_pose_hessian, int dim);
void free_FlexPoseHessian_gpu(FlexPoseHessian* flex_pose_hessian);


/**
 * @brief Atomic force vectors
 */
struct FlexForce{
    Real* f; // size: natom * 3
};
FlexForce* alloc_FlexForce_gpu(int natom);
FlexForce* alloccp_FlexForce_gpu(const FlexForce& flex_force, int natom);
void free_FlexForce_gpu(FlexForce* flex_force);


/**
 * @brief Common topology. Docking logic
 */
struct FlexTopo{
    int natom = 0;
    int ntorsion = 0;

    int* vn_types; // size: natom. namely saving all elements

    int* axis_atoms; // size: ntorsion * 2. num_torsion rotation axis, each two is (from, to)

    int* range_inds; // size: ntorsion * 2. [start_index, num_range] for each torsion
    Real* range_list; // size: range_inds[ntorsion * 2] + range_inds[ntorsion * 2 + 1] + 1. from torsion library [-0.8pi,-0.7pi], [0.3pi, 0.9pi] ...

    int* rotated_inds; // size: ntorsion * 2. [start_index, end_index] for each torsion, saving index to find rotated_atoms
    int* rotated_atoms; // size: rotated_inds[ntorsion * 2 + 1] + 1. rotated atoms list
};


FlexTopo* alloccp_FlexTopo_gpu(const FlexTopo& flex_topo);
void free_FlexTopo_gpu(FlexTopo* flex_topo);


/**
 * @brief FixMol has no pose, since its pose is itself.
 */
struct FixMol{
    int natom = 0;
    Real* coords;
};
// only serves for ONE object
FixMol* alloccp_FixMol_gpu(const FixMol& fix_mol);
void free_FixMol_gpu(FixMol* fix_mol_cu);


enum ScoreFunc{vina, gaff2};
const std::array<std::string, 2> SCOREFUNC_NAMES = {"vina", "gaff2"};


struct Box{ 
    // Box is the docking area.
    // Ligand atoms are not allowed to move out of the box.
    Real x_lo = -30; // low bound of x.
    Real x_hi = 30; // high bound of x.
    Real y_lo = -30; // low bound of y
    Real y_hi = 30; // high bound of y
    Real z_lo = -30; // low bound of z
    Real z_hi = 30; // high bound of z
    SCOPE_INLINE bool is_inside(Real x, Real y, Real z) const{
        return (x_lo <= x && x <= x_hi) && (y_lo <= y && y <= y_hi) && (z_lo <= z && z <= z_hi);
    }
};

enum BiasType{
    BT_NO,
    BT_POS,
    BT_ALIGN
};

struct DockParam{
    int seed = 12345;
    bool constraint_docking = false;
    int exhaustiveness = 128;
    bool randomize = true;
    int mc_steps = 20; // MC steps
    int opt_steps = -1; // optimization steps in MC process. Zero if only pure MC search is required.
    int refine_steps = 5; // optimization steps after MC, namely a pure local refinement
    int num_pose = 10;
    Real energy_range = 10.0;
    Real rmsd_limit = 1.0; // a limit to judge whether two poses are the same during clustering

    ScoreFunc search_score = vina;
    ScoreFunc opt_score = vina;
    Box box;
    BiasType bias_type;
    Real bias_k = 0.1;
    DockParam() = default;

    void show() const{
        spdlog::info("DockParam: seed={}, search_score={}, opt_score={}, \n"
                     "box: x_lo={} Angstrom, x_hi={} Angstrom, y_lo={} Angstrom, y_hi={} Angstrom, z_lo={} Angstrom, z_hi={} Angstrom, \n"
                     "constraint_docking={}, exhaustiveness={}, \n"
                     "mc_steps={}, opt_steps={}, refine_steps={}, \n"
                     "num_pose={}, rmsd_limit={} Angstrom, \n"
                     "bias_type={}, bias_k={}\n",
                     seed, static_cast<int>(search_score), static_cast<int>(opt_score),
                     box.x_lo, box.x_hi, box.y_lo, box.y_hi, box.z_lo, box.z_hi,
                     constraint_docking, exhaustiveness,
                     mc_steps, opt_steps, refine_steps,
                     num_pose, rmsd_limit,
                     int(bias_type), bias_k);
    }
};

//------------------------------  Pre- and Post-process ------------------------------
// Use UDxxx Models, which are redundant models.

/**
 * @brief Redundant Data Model for rotatable torsions.
 */
struct UDTorsion{
    int axis[2] = {-1}; // rotation axis atoms
    int atoms[4] = {-1};
    std::vector<int> rotated_atoms; // all atoms rotated by this torsion
    std::vector<Real> range_list; // each two is v_lo, v_
    std::vector<Real> param_gaff2; // each 4 elements are one group 
};


struct Bias{
    int i;
    float param[5]; // [x, y, z, V_set, r]
};

/**
 * @brief Redundant Data Model for flexible molecule, including the ligand and those flexible chains connected to it.
 * Molecule is a natural physical concept. so interactions should not be saved here.
 */
struct UDFlexMol{
    int natom = 0;
    Real center[3] = {0};
    std::string name;
    std::vector<Real> coords;
    std::vector<int> vina_types; // vina in default todo?
    std::vector<int> ff_types;
    std::vector<Real> charges;
    std::set<std::pair<int, int>> pairs_1213; // ignored pairs in vdw summation
    std::set<std::pair<int, int>> pairs_14; // ignored torsions in dihedral calculation
    std::vector<UDTorsion> torsions;
    std::vector<Real> dihedrals;
    std::vector<int> intra_pairs; // each in ligand x each in ligand, except 1-2, 1-3, 1-4
    std::vector<int> inter_pairs; // each in ligand x each in protein (they both count their own atoms from zero)
    std::vector<Real> r1_plus_r2_intra; // vdW radii summations of each pair, intra part
    std::vector<Real> r1_plus_r2_inter; // vdW radii summations of each pair, inter part
    std::vector<Bias> biases;
};

typedef std::vector<UDFlexMol> UDFlexMolList;


/**
 * @brief Redundant Data Model for fixed molecule.
 */
struct UDFixMol{
    int natom; // only records those inside box
    std::vector<Real> coords;
    std::vector<int> vina_types; // vina in default todo?
    std::vector<int> ff_types;
    std::vector<Real> charges;
};


#endif //MOLECULE_H
