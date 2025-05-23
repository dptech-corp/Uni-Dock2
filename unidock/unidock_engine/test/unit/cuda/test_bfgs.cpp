//
// Created by Congcong Liu on 24-12-9.
//

#include <algorithm>
#include <catch2/catch_amalgamated.hpp>

#include "geometry/rotation.h"
#include "optimize/bfgs.h"


void prepare_param_by_topo(const FlexTopo* flex_topo, const FixMol* fix_mol, const FixParamVina* fix_param, FlexParamVina* flex_param){
    flex_param->npair_intra = flex_topo->natom * (flex_topo->natom - 1) / 2;
    if (flex_param->npair_intra > 0){
        flex_param->pairs_intra = new int[flex_param->npair_intra * 2];
        flex_param->r1_plus_r2_intra = new Real[flex_param->npair_intra];
    }else{
        flex_param->pairs_intra = nullptr;
        flex_param->r1_plus_r2_intra = nullptr;
    }
    flex_param->atom_types = new int[flex_topo->natom];
    int ipair = 0;
    for (int i = 0; i < flex_topo->natom; i++){
        for (int j = i + 1; j < flex_topo->natom; j++){
            flex_param->pairs_intra[ipair * 2] = i;
            flex_param->pairs_intra[ipair * 2 + 1] = j;
            flex_param->r1_plus_r2_intra[ipair] = VN_VDW_RADII[flex_topo->vn_types[i]] + VN_VDW_RADII[flex_topo->vn_types[j]];
            ipair++;
        }
        flex_param->atom_types[i] = flex_topo->vn_types[i];
    }

    flex_param->npair_inter = flex_topo->natom * fix_mol->natom;
    flex_param->pairs_inter = new int[flex_param->npair_inter * 2];
    flex_param->r1_plus_r2_inter = new Real[flex_param->npair_inter];
    ipair = 0;
    for (int i = 0; i < flex_topo->natom; i++){
        for (int j = 0; j < fix_mol->natom; j++){
            flex_param->pairs_inter[ipair * 2] = i;
            flex_param->pairs_inter[ipair * 2 + 1] = j;
            flex_param->r1_plus_r2_inter[ipair] = VN_VDW_RADII[flex_param->atom_types[i]] + VN_VDW_RADII[fix_param->atom_types[j]];
            ipair++;
        }
    }
}

void prepare_flex_1(FixMol* fix_mol, FixParamVina* fix_param, FlexParamVina* flex_param,
    FlexPose* x, FlexTopo* flex_topo, FlexPose* out_x_new, FlexPoseGradient* out_g_new,
    Real r = 1.1){

    fix_mol->natom = 1;
    fix_mol->coords = new Real[3];
    fix_mol->coords[0] = 0.;
    fix_mol->coords[1] = 0.;
    fix_mol->coords[2] = 0.;
    fix_param->atom_types = new int[1];
    fix_param->atom_types[0] = VN_TYPE_C_H;

    flex_topo->natom = 1;
    flex_topo->vn_types = new int[1];
    flex_topo->vn_types[0] = VN_TYPE_C_H;

    prepare_param_by_topo(flex_topo, fix_mol, fix_param, flex_param);

    x->energy = 999;
    x->center[0] = r;
    x->center[1] = 0.;
    x->center[2] = 0.;
    x->coords = new Real[3];
    x->coords[0] = r;
    x->coords[1] = 0.;
    x->coords[2] = 0.;
    x->dihedrals = nullptr;

    out_x_new->energy = 999;
    out_x_new->coords = new Real[3];
    std::fill(out_x_new->coords, out_x_new->coords + 3, 0.);
    out_x_new->dihedrals = nullptr;

    out_g_new->dihedrals_g = nullptr;
}




TEST_CASE("test bfgs process", "[bfgs]"){
    Real out_e = 9999, out_alpha = -9999;
    FixMol fix_mol;
    FixParamVina fix_param;
    FlexParamVina flex_param;
    FlexPose x;
    FlexTopo flex_topo;
    FlexPose out_x_new;
    FlexPoseGradient out_g_new;
    Vina vina;
    Real g_ref = 0;
    Real e_ref = 0;

    //--------------- Test 1 -----------------
    // One Carbon as receptor, one Carbon as flex
    for (Real r = 0; r < 2; r += 0.5){
        // Set Receptor
        prepare_flex_1(&fix_mol, &fix_param, &flex_param, &x, &flex_topo, &out_x_new, &out_g_new, r);

        cal_e_grad_one_gpu(fix_mol, fix_param, flex_param, x, flex_topo, &out_g_new, &out_e);
        e_ref = vina.eval_ef(r - flex_param.r1_plus_r2_inter[0], fix_param.atom_types[0], flex_param.atom_types[0], &g_ref);
        REQUIRE_THAT(out_e, Catch::Matchers::WithinAbs(e_ref, 1e-4));
        if (r == 0.){
            g_ref = 0.; // too close, the calculation generates wrong result
        }
        REQUIRE_THAT(out_g_new.center_g[0], Catch::Matchers::WithinAbs(g_ref, 1e-4));
        REQUIRE_THAT(out_g_new.center_g[1], Catch::Matchers::WithinAbs(0, 1e-4));
        REQUIRE_THAT(out_g_new.center_g[2], Catch::Matchers::WithinAbs(0, 1e-4));
        REQUIRE_THAT(out_g_new.orientation_g[0], Catch::Matchers::WithinAbs(0, 1e-4));
        REQUIRE_THAT(out_g_new.orientation_g[1], Catch::Matchers::WithinAbs(0, 1e-4));
        REQUIRE_THAT(out_g_new.orientation_g[2], Catch::Matchers::WithinAbs(0, 1e-4));
    }

    // r == 1.5, record the g
    Real g[3] = { -out_g_new.center_g[0], -out_g_new.center_g[1], -out_g_new.center_g[2] };
    line_search_one_gpu(fix_mol, fix_param,
        flex_param, x,
        flex_topo,
        &out_x_new, &out_g_new, &out_e, &out_alpha);

    REQUIRE_THAT(out_alpha, Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(out_x_new.coords[0], Catch::Matchers::WithinAbs(x.coords[0] + out_alpha * g[0], 1e-4));
    REQUIRE_THAT(out_x_new.coords[1], Catch::Matchers::WithinAbs(x.coords[1] + out_alpha * g[1], 1e-4));
    REQUIRE_THAT(out_x_new.coords[2], Catch::Matchers::WithinAbs(x.coords[2] + out_alpha * g[2], 1e-4));
    Real r = cal_norm(out_x_new.coords);
    REQUIRE_THAT(out_e, Catch::Matchers::WithinAbs(vina.eval_ef(r - flex_param.r1_plus_r2_inter[0], fix_param.atom_types[0], flex_param.atom_types[0], &g_ref), 1e-4));
    //![bfgs]
}


void prepare_flex_2(FixMol* fix_mol, FixParamVina* fix_param, FlexParamVina* flex_param,
    FlexPose* x, FlexTopo* flex_topo, FlexPose* out_x_new, FlexPoseGradient* out_g_new,
    const Real c1[3], const Real c2[3]){
    // set fix
    fix_mol->natom = 1;
    fix_mol->coords = new Real[3];
    fix_mol->coords[0] = 0.;
    fix_mol->coords[1] = 0.;
    fix_mol->coords[2] = 0.;

    fix_param->atom_types = new int[1];
    fix_param->atom_types[0] = VN_TYPE_C_H;

    // set flex
    flex_topo->natom = 2;
    flex_topo->vn_types = new int[2];
    flex_topo->vn_types[0] = VN_TYPE_C_H;
    flex_topo->vn_types[1] = VN_TYPE_C_H;

    prepare_param_by_topo(flex_topo, fix_mol, fix_param, flex_param);

    x->energy = 999;
    x->coords = new Real[6];
    x->coords[0] = c1[0];
    x->coords[1] = c1[1];
    x->coords[2] = c1[2];
    x->coords[3] = c2[0];
    x->coords[4] = c2[1];
    x->coords[5] = c2[2];
    x->dihedrals = nullptr;

    x->center[0] = (c1[0] + c2[0]) / 2.;
    x->center[1] = (c1[1] + c2[1]) / 2.;
    x->center[2] = (c1[2] + c2[2]) / 2.;

    out_x_new->energy = 999;
    out_x_new->coords = new Real[6];
    std::fill(out_x_new->coords, out_x_new->coords + 6, 0.);
    out_x_new->dihedrals = nullptr;

    out_g_new->dihedrals_g = nullptr;
}

void rotate_x(FlexPose &out_x, const FlexPose &x, std::array<Real, 3> &rot_vec, int natom){
    Real angle_axis[4];
    rotvec_to_axis_angle(angle_axis, rot_vec.data());
    for (int j = 0; j < natom; ++j){
        out_x.coords[j * 3] = x.coords[j * 3] - x.center[0];
        out_x.coords[j * 3 + 1] = x.coords[j * 3 + 1] - x.center[1];
        out_x.coords[j * 3 + 2] = x.coords[j * 3 + 2] - x.center[2];
        rotate_vec_by_rodrigues(out_x.coords + j * 3, angle_axis + 1, angle_axis[0]);
        out_x.coords[j * 3] += x.center[0];
        out_x.coords[j * 3 + 1] += x.center[1];
        out_x.coords[j * 3 + 2] += x.center[2];
    }
    out_x.rot_vec[0] = rot_vec[0];
    out_x.rot_vec[1] = rot_vec[1];
    out_x.rot_vec[2] = rot_vec[2];
}

void delete_flex_2(FixMol* fix_mol, FixParamVina* fix_param,
    FlexPose* x, FlexTopo* flex_topo, FlexPose* out_x_new){
    delete fix_mol->coords;
    fix_mol->coords = nullptr;

    delete fix_param->atom_types;
    fix_param->atom_types = nullptr;

    delete flex_topo->vn_types;
    flex_topo->vn_types = nullptr;

    delete x->coords;
    x->coords = nullptr;

    delete out_x_new->coords;
    out_x_new->coords = nullptr;
}


TEST_CASE("test cal_e_grad_one", "[cal_e_grad_one]"){
    // simple and quick, without memory free
    Real out_e_1 = 0;
    Real out_e_2 = 0;

    FixMol fix_mol;
    FixParamVina fix_param;
    FlexParamVina flex_param;
    FlexPose x;
    FlexTopo flex_topo;
    FlexPose out_x_new;
    FlexPoseGradient out_g_1;
    FlexPoseGradient out_g_2;
    Vina vina;

    //--------------- Prepare Input Data -----------------
    // One Carbon as receptor, two Carbons as flex
    Real r = 1;
    // Set Receptor
    std::array<Real, 3> ca = {r, 0, 0};
    std::array<Real, 3> cb = {r, 1, 0};
    prepare_flex_2(&fix_mol, &fix_param, &flex_param, &x, &flex_topo, &out_x_new, &out_g_1, ca.data(), cb.data());
    cal_e_grad_one_gpu(fix_mol, fix_param, flex_param, x, flex_topo, &out_g_1, &out_e_1);


    //--------------- Check Energy -----------------
    Real e_ref =0, g_tmp = 0;
    e_ref += vina.eval_ef(r - flex_param.r1_plus_r2_inter[0], fix_param.atom_types[0], flex_param.atom_types[0], &g_tmp);
    e_ref += vina.eval_ef(sqrt(r * r + 1) - flex_param.r1_plus_r2_inter[1], fix_param.atom_types[0], flex_param.atom_types[1], &g_tmp);
    e_ref += vina.eval_ef(1 - flex_param.r1_plus_r2_intra[0], flex_param.atom_types[0], flex_param.atom_types[1], &g_tmp);
    REQUIRE_THAT(out_e_1, Catch::Matchers::WithinAbs(e_ref, 1e-4));
    delete_flex_2(&fix_mol, &fix_param, &x, &flex_topo, &out_x_new);

    //--------------- Check Gradient (Numerical Method) -----------------
    // 1. Position Gradient
    Real dr = 1e-3;
    Real g_num = 0.;
    for (int i = 0; i < 3; ++i){ // each direction: x, y, z
        auto ca_2 = ca;
        ca_2[i] += dr;
        auto cb_2 = cb;
        cb_2[i] += dr;

        prepare_flex_2(&fix_mol, &fix_param, &flex_param, &x, &flex_topo, &out_x_new, &out_g_2, ca_2.data(), cb_2.data());
        // compute energy
        cal_e_grad_one_gpu(fix_mol, fix_param, flex_param, x, flex_topo, &out_g_2, &out_e_2);
        g_num = (out_e_2 - out_e_1) / dr;
        REQUIRE_THAT(out_g_1.center_g[i], Catch::Matchers::WithinAbs(g_num, 0.005));
        delete_flex_2(&fix_mol, &fix_param, &x, &flex_topo, &out_x_new);
    }

    // 2. Rotation Gradient
    std::array<Real, 3> rot_vec = {-1.1, 0.2, 0.3}; //; // {0, 0., 0.} todo: check theta out of [-2pi, 2pi]

    FlexPose x_2;
    /// the initial pose, not rotated
    prepare_flex_2(&fix_mol, &fix_param, &flex_param, &x, &flex_topo, &out_x_new, &out_g_1, ca.data(), cb.data());
    printf("Original coords: %f, %f, %f; %f, %f, %f\n",
        x.coords[0], x.coords[1], x.coords[2], x.coords[3], x.coords[4], x.coords[5]);
    /// first, rotated by rot_vec
    prepare_flex_2(&fix_mol, &fix_param, &flex_param, &x_2, &flex_topo, &out_x_new, &out_g_1, ca.data(), cb.data());
    rotate_x(x_2, x, rot_vec, flex_topo.natom);
    // the e and g of 1st rotation, out_g_1 is recorded
    cal_e_grad_one_gpu(fix_mol, fix_param, flex_param, x_2, flex_topo, &out_g_1, &out_e_1);

    // second, rotated by rot_vec + dr_i
    dr = 0.005;
    for (int i = 0; i < 3; ++i){ // x, y, z in rot vec
        auto v2 = rot_vec;
        v2[i] += dr;
        rotate_x(x_2, x, v2, flex_topo.natom);
        cal_e_grad_one_gpu(fix_mol, fix_param, flex_param, x_2, flex_topo, &out_g_2, &out_e_2);

        g_num = (out_e_2 - out_e_1) / dr;
        REQUIRE_THAT(out_g_1.orientation_g[i], Catch::Matchers::WithinAbs(g_num, 0.01));
    }

    delete_flex_2(&fix_mol, &fix_param, &x_2, &flex_topo, &out_x_new);
    delete_flex_2(&fix_mol, &fix_param, &x, &flex_topo, &out_x_new);

    //![cal_e_grad_one]
}