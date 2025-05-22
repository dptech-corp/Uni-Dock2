//
// Created by Congcong Liu on 24-12-9.
//

#include "model/model.h"

// ==================  Memory allocation for ONE object ==================
FlexPose* alloccp_FlexPose_gpu(const FlexPose& flex_pose, int natom, int ntorsion){
    FlexPose* flex_pose_cu;
    checkCUDA(cudaMalloc(&flex_pose_cu, sizeof(FlexPose)));

    FlexPose flex_pose_tmp; 
    flex_pose_tmp.energy = flex_pose.energy;
    flex_pose_tmp.center[0] = flex_pose.center[0];
    flex_pose_tmp.center[1] = flex_pose.center[1];
    flex_pose_tmp.center[2] = flex_pose.center[2];
    flex_pose_tmp.rot_vec[0] = flex_pose.rot_vec[0];
    flex_pose_tmp.rot_vec[1] = flex_pose.rot_vec[1];
    flex_pose_tmp.rot_vec[2] = flex_pose.rot_vec[2];
    checkCUDA(cudaMalloc(&flex_pose_tmp.coords, sizeof(Real) * natom * 3));
    checkCUDA(cudaMemcpy(flex_pose_tmp.coords, flex_pose.coords, sizeof(Real) * natom * 3, cudaMemcpyHostToDevice));
    checkCUDA(cudaMalloc(&flex_pose_tmp.dihedrals, sizeof(Real) * ntorsion));
    checkCUDA(cudaMemcpy(flex_pose_tmp.dihedrals, flex_pose.dihedrals, sizeof(Real) * ntorsion, cudaMemcpyHostToDevice));

    checkCUDA(cudaMemcpy(flex_pose_cu, &flex_pose_tmp, sizeof(FlexPose), cudaMemcpyHostToDevice));
    return flex_pose_cu;
}   
void free_FlexPose_gpu(FlexPose* flex_pose_cu){
    FlexPose flex_pose_tmp;
    checkCUDA(cudaMemcpy(&flex_pose_tmp, flex_pose_cu, sizeof(FlexPose), cudaMemcpyDeviceToHost));
    checkCUDA(cudaFree(flex_pose_tmp.coords));
    checkCUDA(cudaFree(flex_pose_tmp.dihedrals));
    checkCUDA(cudaFree(flex_pose_cu));
}
void freecp_FlexPose_gpu(FlexPose* flex_pose_cu, FlexPose* out_flex_pose, int natom, int ntorsion){
    Real* p1 = out_flex_pose->coords;
    Real* p2 = out_flex_pose->dihedrals;
    checkCUDA(cudaMemcpy(out_flex_pose, flex_pose_cu, sizeof(FlexPose), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(p1, out_flex_pose->coords, sizeof(Real) * natom * 3, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(p2, out_flex_pose->dihedrals, sizeof(Real) * ntorsion, cudaMemcpyDeviceToHost));
    checkCUDA(cudaFree(out_flex_pose->coords));
    checkCUDA(cudaFree(out_flex_pose->dihedrals));
    checkCUDA(cudaFree(flex_pose_cu));
    out_flex_pose->coords = p1;
    out_flex_pose->dihedrals = p2;
}



FlexPoseGradient* alloccp_FlexPoseGradient_gpu(const FlexPoseGradient& flex_pose_gradient, int ntorsion){
    FlexPoseGradient* flex_pose_gradient_cu;
    checkCUDA(cudaMalloc(&flex_pose_gradient_cu, sizeof(FlexPoseGradient)));

    FlexPoseGradient flex_pose_gradient_tmp;
    flex_pose_gradient_tmp.center_g[0] = flex_pose_gradient.center_g[0];
    flex_pose_gradient_tmp.center_g[1] = flex_pose_gradient.center_g[1];
    flex_pose_gradient_tmp.center_g[2] = flex_pose_gradient.center_g[2];
    flex_pose_gradient_tmp.orientation_g[0] = flex_pose_gradient.orientation_g[0];
    flex_pose_gradient_tmp.orientation_g[1] = flex_pose_gradient.orientation_g[1];
    flex_pose_gradient_tmp.orientation_g[2] = flex_pose_gradient.orientation_g[2];

    checkCUDA(cudaMalloc(&flex_pose_gradient_tmp.dihedrals_g, sizeof(Real) * ntorsion));
    checkCUDA(cudaMemcpy(flex_pose_gradient_tmp.dihedrals_g, flex_pose_gradient.dihedrals_g, sizeof(Real) * ntorsion, cudaMemcpyHostToDevice));
    
    checkCUDA(cudaMemcpy(flex_pose_gradient_cu, &flex_pose_gradient_tmp, sizeof(FlexPoseGradient), cudaMemcpyHostToDevice));
    return flex_pose_gradient_cu;
}
void free_FlexPoseGradient_gpu(FlexPoseGradient* flex_pose_gradient_cu){
    FlexPoseGradient flex_pose_gradient_tmp;
    checkCUDA(cudaMemcpy(&flex_pose_gradient_tmp, flex_pose_gradient_cu, sizeof(FlexPoseGradient), cudaMemcpyDeviceToHost));
    checkCUDA(cudaFree(flex_pose_gradient_tmp.dihedrals_g));
    checkCUDA(cudaFree(flex_pose_gradient_cu));
}
void freecp_FlexPoseGradient_gpu(FlexPoseGradient* flex_pose_gradient_cu, FlexPoseGradient* out_flex_pose_gradient, int ntorsion){
    Real* p1 = out_flex_pose_gradient->dihedrals_g; 
    checkCUDA(cudaMemcpy(out_flex_pose_gradient, flex_pose_gradient_cu, sizeof(FlexPoseGradient), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(p1, out_flex_pose_gradient->dihedrals_g, sizeof(Real) * ntorsion, cudaMemcpyDeviceToHost));
    checkCUDA(cudaFree(out_flex_pose_gradient->dihedrals_g));
    checkCUDA(cudaFree(flex_pose_gradient_cu));
    out_flex_pose_gradient->dihedrals_g = p1;
}



FlexPoseHessian* alloc_FlexPoseHessian_gpu(int ntorsion){
    int dim = 3 + 4 + ntorsion;
    FlexPoseHessian* flex_pose_hessian_cu;
    checkCUDA(cudaMalloc(&flex_pose_hessian_cu, sizeof(FlexPoseHessian)));

    FlexPoseHessian flex_pose_hessian_tmp;
    checkCUDA(cudaMalloc(&flex_pose_hessian_tmp.matrix, sizeof(Real) * dim * (dim + 1) / 2));

    checkCUDA(cudaMemcpy(flex_pose_hessian_cu, &flex_pose_hessian_tmp, sizeof(FlexPoseHessian), cudaMemcpyHostToDevice));
    return flex_pose_hessian_cu;
}
FlexPoseHessian* alloccp_FlexPoseHessian_gpu(const FlexPoseHessian& flex_pose_hessian, int ntorsion){
    int dim = 3 + 4 + ntorsion;
    FlexPoseHessian* flex_pose_hessian_cu;
    checkCUDA(cudaMalloc(&flex_pose_hessian_cu, sizeof(FlexPoseHessian)));

    FlexPoseHessian flex_pose_hessian_tmp;
    checkCUDA(cudaMalloc(&flex_pose_hessian_tmp.matrix, sizeof(Real) * dim * (dim + 1) / 2));
    checkCUDA(cudaMemcpy(flex_pose_hessian_tmp.matrix, flex_pose_hessian.matrix, sizeof(Real) * dim * (dim + 1) / 2, cudaMemcpyHostToDevice));

    checkCUDA(cudaMemcpy(flex_pose_hessian_cu, &flex_pose_hessian_tmp, sizeof(FlexPoseHessian), cudaMemcpyHostToDevice));
    return flex_pose_hessian_cu;
}   
void free_FlexPoseHessian_gpu(FlexPoseHessian* flex_pose_hessian_cu){
    FlexPoseHessian flex_pose_hessian_tmp;  
    checkCUDA(cudaMemcpy(&flex_pose_hessian_tmp, flex_pose_hessian_cu, sizeof(FlexPoseHessian), cudaMemcpyDeviceToHost));
    checkCUDA(cudaFree(flex_pose_hessian_tmp.matrix));
    checkCUDA(cudaFree(flex_pose_hessian_cu));
}


FlexForce* alloc_FlexForce_gpu(int natom){
    FlexForce* flex_force_cu;
    checkCUDA(cudaMalloc(&flex_force_cu, sizeof(FlexForce)));

    FlexForce flex_force_tmp;
    checkCUDA(cudaMalloc(&flex_force_tmp.f, sizeof(Real) * natom * 3));

    checkCUDA(cudaMemcpy(flex_force_cu, &flex_force_tmp, sizeof(FlexForce), cudaMemcpyHostToDevice));
    return flex_force_cu;
}
FlexForce* alloccp_FlexForce_gpu(const FlexForce& flex_force, int natom){
    FlexForce* flex_force_cu;
    checkCUDA(cudaMalloc(&flex_force_cu, sizeof(FlexForce)));

    FlexForce flex_force_tmp;
    checkCUDA(cudaMalloc(&flex_force_tmp.f, sizeof(Real) * natom * 3));
    checkCUDA(cudaMemcpy(flex_force_tmp.f, flex_force.f, sizeof(Real) * natom * 3, cudaMemcpyHostToDevice));

    checkCUDA(cudaMemcpy(flex_force_cu, &flex_force_tmp, sizeof(FlexForce), cudaMemcpyHostToDevice));
    return flex_force_cu;
}
void free_FlexForce_gpu(FlexForce* flex_force_cu){
    FlexForce flex_force_tmp;
    checkCUDA(cudaMemcpy(&flex_force_tmp, flex_force_cu, sizeof(FlexForce), cudaMemcpyDeviceToHost));   
    checkCUDA(cudaFree(flex_force_tmp.f));
    checkCUDA(cudaFree(flex_force_cu));
}



FlexTopo* alloccp_FlexTopo_gpu(const FlexTopo& flex_topo){
    FlexTopo* flex_topo_cu;
    checkCUDA(cudaMalloc(&flex_topo_cu, sizeof(FlexTopo)));
    
    FlexTopo flex_topo_tmp;
    flex_topo_tmp.natom = flex_topo.natom;
    flex_topo_tmp.ntorsion = flex_topo.ntorsion;
    // Allocate memory for each pointer member on the GPU
    checkCUDA(cudaMalloc(&flex_topo_tmp.vn_types, sizeof(int) * flex_topo.natom));
    checkCUDA(cudaMemcpy(flex_topo_tmp.vn_types, flex_topo.vn_types, sizeof(int) * flex_topo.natom, cudaMemcpyHostToDevice));
    checkCUDA(cudaMalloc(&flex_topo_tmp.axis_atoms, sizeof(int) * flex_topo.ntorsion * 2));
    checkCUDA(cudaMemcpy(flex_topo_tmp.axis_atoms, flex_topo.axis_atoms, sizeof(int) * flex_topo.ntorsion * 2, cudaMemcpyHostToDevice));
    checkCUDA(cudaMalloc(&flex_topo_tmp.range_inds, sizeof(int) * flex_topo.ntorsion * 2));
    checkCUDA(cudaMemcpy(flex_topo_tmp.range_inds, flex_topo.range_inds, sizeof(int) * flex_topo.ntorsion * 2, cudaMemcpyHostToDevice));

    int range_list_size = flex_topo.ntorsion > 0 ? flex_topo.range_inds[flex_topo.ntorsion * 2] + flex_topo.range_inds[flex_topo.ntorsion * 2 + 1] + 1 : 0;
    checkCUDA(cudaMalloc(&flex_topo_tmp.range_list, sizeof(Real) * range_list_size)); // Assuming each torsion has two range values
    checkCUDA(cudaMemcpy(flex_topo_tmp.range_list, flex_topo.range_list, sizeof(Real) * range_list_size, cudaMemcpyHostToDevice));

    checkCUDA(cudaMalloc(&flex_topo_tmp.rotated_inds, sizeof(int) * flex_topo.ntorsion * 2));
    checkCUDA(cudaMemcpy(flex_topo_tmp.rotated_inds, flex_topo.rotated_inds, sizeof(int) * flex_topo.ntorsion * 2, cudaMemcpyHostToDevice));

    int rotated_atoms_size = flex_topo.ntorsion > 0 ? flex_topo.rotated_inds[flex_topo.ntorsion * 2 + 1] + 1 : 0;
    checkCUDA(cudaMalloc(&flex_topo_tmp.rotated_atoms, sizeof(int) * rotated_atoms_size)); // Assuming maximum size
    checkCUDA(cudaMemcpy(flex_topo_tmp.rotated_atoms, flex_topo.rotated_atoms, sizeof(int) * rotated_atoms_size, cudaMemcpyHostToDevice));

    checkCUDA(cudaMemcpy(flex_topo_cu, &flex_topo_tmp, sizeof(FlexTopo), cudaMemcpyHostToDevice));

    return flex_topo_cu;
}
void free_FlexTopo_gpu(FlexTopo* flex_topo_cu){
    FlexTopo flex_topo_tmp;
    checkCUDA(cudaMemcpy(&flex_topo_tmp, flex_topo_cu, sizeof(FlexTopo), cudaMemcpyDeviceToHost));
    checkCUDA(cudaFree(flex_topo_tmp.vn_types));
    checkCUDA(cudaFree(flex_topo_tmp.axis_atoms));
    checkCUDA(cudaFree(flex_topo_tmp.range_inds));
    checkCUDA(cudaFree(flex_topo_tmp.range_list));
    checkCUDA(cudaFree(flex_topo_tmp.rotated_inds));
    checkCUDA(cudaFree(flex_topo_tmp.rotated_atoms));
    checkCUDA(cudaFree(flex_topo_cu));
}


FixMol* alloccp_FixMol_gpu(const FixMol& fix_mol){
    FixMol* fix_mol_cu;
    checkCUDA(cudaMalloc(&fix_mol_cu, sizeof(FixMol)));

    FixMol fix_mol_tmp;
    fix_mol_tmp.natom = fix_mol.natom;
    checkCUDA(cudaMalloc(&fix_mol_tmp.coords, sizeof(Real) * fix_mol.natom * 3));
    checkCUDA(cudaMemcpy(fix_mol_tmp.coords, fix_mol.coords, sizeof(Real) * fix_mol.natom * 3, cudaMemcpyHostToDevice));

    checkCUDA(cudaMemcpy(fix_mol_cu, &fix_mol_tmp, sizeof(FixMol), cudaMemcpyHostToDevice));
    return fix_mol_cu;
}

void free_FixMol_gpu(FixMol* fix_mol_cu){
    FixMol fix_mol_tmp;
    checkCUDA(cudaMemcpy(&fix_mol_tmp, fix_mol_cu, sizeof(FixMol), cudaMemcpyDeviceToHost));
    checkCUDA(cudaFree(fix_mol_tmp.coords));
    checkCUDA(cudaFree(fix_mol_cu));
}







