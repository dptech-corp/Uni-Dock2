//
// Created by Congcong Liu on 24-11-13.
//

#include <cstring>
#include <iterator>
#include <algorithm>
#include <vector>
#include "common.cuh"
#include "task/DockTask.h"
#include "myutils/errors.h"
#include "cuda/struct_array_manager.cuh"
#include "model/model.h"


static void alloc_flex_topo_list(StructArrayManager<FlexTopo>*& flex_topo_list_manager, FlexTopo*& flex_topo_list_cu,
                          const UDFlexMolList& udflex_mols,
                          const std::vector<int>& list_n_atom_flex, const std::vector<int>& list_n_dihe,
                          const std::vector<int>& list_n_range, const std::vector<int>& list_n_rotated_atoms){

    int nflex = udflex_mols.size();
    flex_topo_list_manager = new StructArrayManager<FlexTopo>(nflex);

    std::vector<int> list_n_dihe_x2(nflex);
    std::transform(list_n_dihe.begin(), list_n_dihe.end(), list_n_dihe_x2.begin(), [](int v){ return v * 2; });
    std::vector<int> list_range_list_size(nflex);
    std::transform(list_n_range.begin(), list_n_range.end(), list_range_list_size.begin(), [](int v){ return v * 2; });

    flex_topo_list_manager->add_ptr_field<int*>(StructsMemberPtrField<FlexTopo, int*>{&FlexTopo::vn_types, sizeof(int), list_n_atom_flex});
    flex_topo_list_manager->add_ptr_field<int*>(StructsMemberPtrField<FlexTopo, int*>{&FlexTopo::axis_atoms, sizeof(int), list_n_dihe_x2});
    flex_topo_list_manager->add_ptr_field<int*>(StructsMemberPtrField<FlexTopo, int*>{&FlexTopo::range_inds, sizeof(int), list_n_dihe_x2});
    flex_topo_list_manager->add_ptr_field<int*>(StructsMemberPtrField<FlexTopo, int*>{&FlexTopo::rotated_inds, sizeof(int), list_n_dihe_x2});
    flex_topo_list_manager->add_ptr_field<int*>(StructsMemberPtrField<FlexTopo, int*>{&FlexTopo::rotated_atoms, sizeof(int), list_n_rotated_atoms});
    flex_topo_list_manager->add_ptr_field<Real*>(StructsMemberPtrField<FlexTopo, Real*>{&FlexTopo::range_list, sizeof(Real), list_range_list_size});

    flex_topo_list_manager->allocate_and_assign();

    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];
        auto& flex_topo = flex_topo_list_manager->array_host[i];
        flex_topo.natom = list_n_atom_flex[i];
        flex_topo.ntorsion = list_n_dihe[i];
        std::memcpy(flex_topo.vn_types, m.vina_types.data(), m.natom * sizeof(int));
        int ind_range = 0, ind_rotated = 0;
        for (int j = 0; j < (int)m.torsions.size(); j++){
            std::memcpy(flex_topo.axis_atoms + j * 2, m.torsions[j].axis, 2 * sizeof(int));
            flex_topo.range_inds[j * 2] = ind_range;
            int n = m.torsions[j].range_list.size();
            std::memcpy(flex_topo.range_list + ind_range, m.torsions[j].range_list.data(), n * sizeof(Real));
            ind_range += n;
            int nrange = n / 2;
            flex_topo.range_inds[j * 2 + 1] = nrange;
            flex_topo.rotated_inds[j * 2] = ind_rotated;
            std::memcpy(flex_topo.rotated_atoms + ind_rotated, m.torsions[j].rotated_atoms.data(),
                        m.torsions[j].rotated_atoms.size() * sizeof(int));
            ind_rotated += (int)m.torsions[j].rotated_atoms.size() - 1;
            flex_topo.rotated_inds[j * 2 + 1] = ind_rotated;
            ind_rotated += 1;
        }
    }

    flex_topo_list_manager->copy_to_gpu();
    flex_topo_list_cu = flex_topo_list_manager->array_device;
}


static void alloc_flex_pose_list(StructArrayManager<FlexPose>*& flex_pose_list_manager,
                             FlexPose*& flex_pose_list_cu,
                             std::vector<int>* list_i_real,
                             const UDFlexMolList& udflex_mols, int npose, int nflex, int exhaustiveness,
                             const std::vector<int>& list_natom_flex, const std::vector<int>& list_ndihe){

    flex_pose_list_manager = new StructArrayManager<FlexPose>(npose);

    // 计算每个 FlexPose 需要的大小
    std::vector<int> list_coords_size(npose);
    std::vector<int> list_dihedrals_size(npose);

    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < exhaustiveness; j++){
            int idx = i * exhaustiveness + j;
            list_coords_size[idx] = list_natom_flex[i] * 3;
            list_dihedrals_size[idx] = list_ndihe[i];
        }
    }

    // 添加指针字段
    flex_pose_list_manager->add_ptr_field<Real*>(StructsMemberPtrField<FlexPose, Real*>{&FlexPose::coords, sizeof(Real), list_coords_size});
    flex_pose_list_manager->add_ptr_field<Real*>(StructsMemberPtrField<FlexPose, Real*>{&FlexPose::dihedrals, sizeof(Real), list_dihedrals_size});

    flex_pose_list_manager->allocate_and_assign();

    // 填充数据并记录索引
    list_i_real->push_back(0);
    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];
        for (int j = 0; j < exhaustiveness; j++){
            int idx = i * exhaustiveness + j;
            auto& flex_pose = flex_pose_list_manager->array_host[idx];

            // 设置初始位姿
            flex_pose.center[0] = m.center[0];
            flex_pose.center[1] = m.center[1];
            flex_pose.center[2] = m.center[2];
            flex_pose.rot_vec[0] = 0;
            flex_pose.rot_vec[1] = 0;
            flex_pose.rot_vec[2] = 0;
            flex_pose.energy = 999;

            // 复制坐标和二面角数据
            std::memcpy(flex_pose.coords, m.coords.data(), m.coords.size() * sizeof(Real));
            std::memcpy(flex_pose.dihedrals, m.dihedrals.data(), m.dihedrals.size() * sizeof(Real));

            // 记录索引信息
            list_i_real->push_back(list_i_real->back() + list_natom_flex[i] * 3);
            list_i_real->push_back(list_i_real->back() + list_ndihe[i]);
        }
    }

    flex_pose_list_manager->copy_to_gpu();
    flex_pose_list_cu = flex_pose_list_manager->array_device;
}

static void alloc_flex_param_list(StructArrayManager<FlexParamVina>*& flex_param_list_manager,
                                  FlexParamVina*& flex_param_list_cu,
                                  const UDFlexMolList& udflex_mols,
                                  int size_intra_all_flex, int size_inter_all_flex, int n_atom_all_flex){
    int nflex = udflex_mols.size();
    flex_param_list_manager = new StructArrayManager<FlexParamVina>(nflex);

    // 计算每个 FlexParamVina 需要的大小
    std::vector<int> list_pairs_intra_size(nflex);
    std::vector<int> list_pairs_inter_size(nflex);
    std::vector<int> list_r1_plus_r2_intra_size(nflex);
    std::vector<int> list_r1_plus_r2_inter_size(nflex);
    std::vector<int> list_atom_types_size(nflex);

    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];
        list_pairs_intra_size[i] = m.intra_pairs.size();
        list_pairs_inter_size[i] = m.inter_pairs.size();
        list_r1_plus_r2_intra_size[i] = m.r1_plus_r2_intra.size();
        list_r1_plus_r2_inter_size[i] = m.r1_plus_r2_inter.size();
        list_atom_types_size[i] = m.natom;
    }

    // 添加指针字段
    flex_param_list_manager->add_ptr_field<int*>(StructsMemberPtrField<FlexParamVina, int*>{&FlexParamVina::pairs_intra, sizeof(int), list_pairs_intra_size});
    flex_param_list_manager->add_ptr_field<int*>(StructsMemberPtrField<FlexParamVina, int*>{&FlexParamVina::pairs_inter, sizeof(int), list_pairs_inter_size});
    flex_param_list_manager->add_ptr_field<Real*>(StructsMemberPtrField<FlexParamVina, Real*>{&FlexParamVina::r1_plus_r2_intra, sizeof(Real), list_r1_plus_r2_intra_size});
    flex_param_list_manager->add_ptr_field<Real*>(StructsMemberPtrField<FlexParamVina, Real*>{&FlexParamVina::r1_plus_r2_inter, sizeof(Real), list_r1_plus_r2_inter_size});
    flex_param_list_manager->add_ptr_field<int*>(StructsMemberPtrField<FlexParamVina, int*>{&FlexParamVina::atom_types, sizeof(int), list_atom_types_size});

    flex_param_list_manager->allocate_and_assign();

    // 填充数据
    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];
        auto& flex_param = flex_param_list_manager->array_host[i];
        
        flex_param.npair_intra = m.intra_pairs.size() / 2;
        flex_param.npair_inter = m.inter_pairs.size() / 2;

        // 复制数据到 host 内存
        std::memcpy(flex_param.pairs_intra, m.intra_pairs.data(), m.intra_pairs.size() * sizeof(int));
        std::memcpy(flex_param.pairs_inter, m.inter_pairs.data(), m.inter_pairs.size() * sizeof(int));
        std::memcpy(flex_param.r1_plus_r2_intra, m.r1_plus_r2_intra.data(), m.r1_plus_r2_intra.size() * sizeof(Real));
        std::memcpy(flex_param.r1_plus_r2_inter, m.r1_plus_r2_inter.data(), m.r1_plus_r2_inter.size() * sizeof(Real));
        std::memcpy(flex_param.atom_types, m.vina_types.data(), m.natom * sizeof(int));
    }

    flex_param_list_manager->copy_to_gpu();
    flex_param_list_cu = flex_param_list_manager->array_device;
}

static void alloc_aux_poses(StructArrayManager<FlexPose>*& aux_poses_manager,
                           FlexPose*& aux_poses_cu,
                           int nflex, int exhaustiveness,
                           const std::vector<int>& list_n_atom_flex,
                           const std::vector<int>& list_n_dihe){
    int npose = nflex * exhaustiveness;
    int total_aux_poses = STRIDE_POSE * npose;
    aux_poses_manager = new StructArrayManager<FlexPose>(total_aux_poses);

    // 计算每个 aux pose 需要的大小
    std::vector<int> list_coords_size(total_aux_poses);
    std::vector<int> list_dihedrals_size(total_aux_poses);

    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < exhaustiveness; j++){
            for (int k = 0; k < STRIDE_POSE; k++){
                int idx = i * STRIDE_POSE * exhaustiveness + j * STRIDE_POSE + k;
                list_coords_size[idx] = list_n_atom_flex[i] * 3;
                list_dihedrals_size[idx] = list_n_dihe[i];
            }
        }
    }

    // 添加指针字段
    aux_poses_manager->add_ptr_field<Real*>(StructsMemberPtrField<FlexPose, Real*>{&FlexPose::coords, sizeof(Real), list_coords_size});
    aux_poses_manager->add_ptr_field<Real*>(StructsMemberPtrField<FlexPose, Real*>{&FlexPose::dihedrals, sizeof(Real), list_dihedrals_size});

    aux_poses_manager->allocate_and_assign();
    aux_poses_manager->copy_to_gpu();
    aux_poses_cu = aux_poses_manager->array_device;
}

static void alloc_aux_grads(StructArrayManager<FlexPoseGradient>*& aux_grads_manager,
                           FlexPoseGradient*& aux_grads_cu,
                           int nflex, int exhaustiveness,
                           const std::vector<int>& list_n_dihe){
    int npose = nflex * exhaustiveness;
    int total_aux_grads = STRIDE_G * npose;
    aux_grads_manager = new StructArrayManager<FlexPoseGradient>(total_aux_grads);

    // 计算每个 aux grad 需要的大小
    std::vector<int> list_dihedrals_g_size(total_aux_grads);

    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < exhaustiveness; j++){
            for (int k = 0; k < STRIDE_G; k++){
                int idx = i * STRIDE_G * exhaustiveness + j * STRIDE_G + k;
                list_dihedrals_g_size[idx] = list_n_dihe[i];
            }
        }
    }

    // 添加指针字段
    aux_grads_manager->add_ptr_field<Real*>(StructsMemberPtrField<FlexPoseGradient, Real*>{&FlexPoseGradient::dihedrals_g, sizeof(Real), list_dihedrals_g_size});

    aux_grads_manager->allocate_and_assign();
    aux_grads_manager->copy_to_gpu();
    aux_grads_cu = aux_grads_manager->array_device;
}

static void alloc_aux_hessians(StructArrayManager<FlexPoseHessian>*& aux_hessians_manager,
                              FlexPoseHessian*& aux_hessians_cu,
                              int nflex, int exhaustiveness,
                              const int* list_ndim_trimat){
    int npose = nflex * exhaustiveness;
    aux_hessians_manager = new StructArrayManager<FlexPoseHessian>(npose);

    // 计算每个 hessian 需要的大小
    std::vector<int> list_matrix_size(npose);

    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < exhaustiveness; j++){
            int idx = i * exhaustiveness + j;
            list_matrix_size[idx] = list_ndim_trimat[i];
        }
    }

    // 添加指针字段
    aux_hessians_manager->add_ptr_field<Real*>(StructsMemberPtrField<FlexPoseHessian, Real*>{&FlexPoseHessian::matrix, sizeof(Real), list_matrix_size});

    aux_hessians_manager->allocate_and_assign();
    aux_hessians_manager->copy_to_gpu();
    aux_hessians_cu = aux_hessians_manager->array_device;
}

static void alloc_aux_forces(StructArrayManager<FlexForce>*& aux_forces_manager,
                            FlexForce*& aux_forces_cu,
                            int nflex, int exhaustiveness,
                            const std::vector<int>& list_n_atom_flex){
    int npose = nflex * exhaustiveness;
    aux_forces_manager = new StructArrayManager<FlexForce>(npose);

    // 计算每个 force 需要的大小
    std::vector<int> list_f_size(npose);

    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < exhaustiveness; j++){
            int idx = i * exhaustiveness + j;
            list_f_size[idx] = list_n_atom_flex[i] * 3;
        }
    }

    // 添加指针字段
    aux_forces_manager->add_ptr_field<Real*>(StructsMemberPtrField<FlexForce, Real*>{&FlexForce::f, sizeof(Real), list_f_size});

    aux_forces_manager->allocate_and_assign();
    aux_forces_manager->copy_to_gpu();
    aux_forces_cu = aux_forces_manager->array_device;
}

static void alloc_fix_mol(FixMol*& fix_mol_cu, Real*& fix_mol_real_cu, const UDFixMol& udfix_mol){
    checkCUDA(cudaMalloc(&fix_mol_cu, sizeof(FixMol)));
    checkCUDA(cudaMalloc(&fix_mol_real_cu, udfix_mol.coords.size() * sizeof(Real)));
    FixMol fix_mol;
    fix_mol.natom = udfix_mol.natom;
    fix_mol.coords = fix_mol_real_cu;
    checkCUDA(
        cudaMemcpy(fix_mol.coords, udfix_mol.coords.data(), udfix_mol.coords.size() * sizeof(Real),
            cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(fix_mol_cu, &fix_mol, sizeof(FixMol), cudaMemcpyHostToDevice));
}


void DockTask::cp_to_cpu(){
    // 使用 array manager 复制数据回 CPU
    flex_pose_list_manager->copy_to_host();
    
    // 为了保持兼容性，仍然分配旧格式的内存并复制数据
    int npose = nflex * dock_param.exhaustiveness;
    checkCUDA(cudaMallocHost(&flex_pose_list_res, npose * sizeof(FlexPose)));
    checkCUDA(
        cudaMallocHost(&flex_pose_list_real_res, dock_param.exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex) *
            sizeof(Real)));
    
    // 复制结构体数组
    std::memcpy(flex_pose_list_res, flex_pose_list_manager->array_host, npose * sizeof(FlexPose));
    
    // 复制连续的 Real 数据
    Real* p_real = flex_pose_list_real_res;
    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            int idx = i * dock_param.exhaustiveness + j;
            auto& flex_pose = flex_pose_list_manager->array_host[idx];
            
            // 复制 coords
            int coords_size = list_i_real[idx * 2 + 1] - list_i_real[idx * 2];
            std::memcpy(p_real, flex_pose.coords, coords_size * sizeof(Real));
            p_real += coords_size;
            
            // 复制 dihedrals
            int dihedrals_size = list_i_real[idx * 2 + 2] - list_i_real[idx * 2 + 1];
            std::memcpy(p_real, flex_pose.dihedrals, dihedrals_size * sizeof(Real));
            p_real += dihedrals_size;
        }
    }
}


void DockTask::alloc_gpu(){
    spdlog::info("Memory Allocation on GPU...");
    checkCUDA(cudaDeviceSynchronize());
    //======================= constants =======================
    init_constants(dock_param);
    //======================= global memory =======================

    // get necessary sizes
    int npose = nflex * dock_param.exhaustiveness;

    n_atom_all_flex = 0;
    n_dihe_all_flex = 0;
    int n_range_all_flex = 0, n_rotated_atoms_all_flex = 0;
    int n_dim_all_flex = 0, n_dim_tri_mat_all_flex = 0;
    int size_inter_all_flex = 0, size_intra_all_flex = 0;

    std::vector<int> list_n_atom_flex(nflex);
    std::vector<int> list_n_dihe(nflex);

    std::vector<int> list_n_range(nflex);
    int list_ndim_trimat[nflex];
    std::vector<int> list_n_rotated_atoms(nflex);

    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];

        size_intra_all_flex += m.intra_pairs.size();
        size_inter_all_flex += m.inter_pairs.size(); // all possible pairs

        list_n_atom_flex[i] = m.natom;
        list_n_dihe[i] = m.dihedrals.size();
        n_atom_all_flex += m.natom;
        n_dihe_all_flex += m.dihedrals.size();

        int dim = 3 + 4 + m.dihedrals.size();
        n_dim_all_flex += dim;
        list_ndim_trimat[i] = dim * (dim + 1) / 2;
        n_dim_tri_mat_all_flex += list_ndim_trimat[i];
        list_n_rotated_atoms[i] = 0;
        list_n_range[i] = 0;
        for (auto t : m.torsions){
            list_n_rotated_atoms[i] += t.rotated_atoms.size();
            list_n_range[i] += t.range_list.size() / 2;
        }
        n_rotated_atoms_all_flex += list_n_rotated_atoms[i];
        n_range_all_flex += list_n_range[i];
    }

    //----- flex_pose_list -----
    alloc_flex_pose_list(flex_pose_list_manager, flex_pose_list_cu, &list_i_real, udflex_mols, npose, nflex,
                            dock_param.exhaustiveness, list_n_atom_flex, list_n_dihe);


    //----- flex_topo_list -----
    alloc_flex_topo_list(flex_topo_list_manager, flex_topo_list_cu,
                         udflex_mols, list_n_atom_flex, list_n_dihe, list_n_range, list_n_rotated_atoms);


    //----- fix_mol -----
    // GPU cost: sizeof(FixMol) + udfix_mol.coords.size() * sizeof(Real)
    alloc_fix_mol(fix_mol_cu, fix_mol_real_cu, udfix_mol);


    //----- flex_param_list_cu -----
    alloc_flex_param_list(flex_param_list_manager, flex_param_list_cu, udflex_mols,
                         size_intra_all_flex, size_inter_all_flex, n_atom_all_flex);


    //----- fix_param_cu -----
    // GPU cost: sizeof(FixParamVina) + udfix_mol.natom * sizeof(int)
    checkCUDA(cudaMalloc(&fix_param_cu, sizeof(FixParamVina)));
    checkCUDA(cudaMalloc(&fix_param_int_cu, udfix_mol.natom * sizeof(int)));
    FixParamVina fix_param;
    fix_param.atom_types = fix_param_int_cu;
    checkCUDA(
        cudaMemcpy(fix_param.atom_types, udfix_mol.vina_types.data(), udfix_mol.natom * sizeof(int),
            cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(fix_param_cu, &fix_param, sizeof(FixParamVina), cudaMemcpyHostToDevice));


    //----- aux_pose_cu -----
    alloc_aux_poses(aux_poses_manager, aux_poses_cu, nflex, dock_param.exhaustiveness,
                    list_n_atom_flex, list_n_dihe);


    //----- aux_grads_cu -----
    alloc_aux_grads(aux_grads_manager, aux_grads_cu, nflex, dock_param.exhaustiveness, list_n_dihe);


    //----- aux_hessians_cu -----
    alloc_aux_hessians(aux_hessians_manager, aux_hessians_cu, nflex, dock_param.exhaustiveness, list_ndim_trimat);


    //----- aux_forces_cu -----
    alloc_aux_forces(aux_forces_manager, aux_forces_cu, nflex, dock_param.exhaustiveness, list_n_atom_flex);


    //----- Clustering -----
    int npair = dock_param.exhaustiveness * (dock_param.exhaustiveness - 1) / 2; // tri-mat with diagonal
    checkCUDA(cudaMalloc(&aux_list_e_cu, npose * sizeof(Real)));
    checkCUDA(cudaMalloc(&aux_list_cluster_cu, nflex * dock_param.exhaustiveness * sizeof(int)));
    checkCUDA(cudaMalloc(&aux_rmsd_ij_cu, nflex * npair * 2 * sizeof(int)));
    checkCUDA(cudaMalloc(&clustered_pose_inds_cu, nflex * dock_param.exhaustiveness * sizeof(int)));

    //so large ...
    std::vector<int> aux_rmsd_ij(nflex * npair * 2, 0); // excluding diagonal line
    for (int i = 0; i < nflex; i++){
        int tmp = 2 * (i * npair);
        int a = 0;
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            for (int k = j + 1; k < dock_param.exhaustiveness; k++){
                aux_rmsd_ij[tmp + a * 2] = j;
                aux_rmsd_ij[tmp + a * 2 + 1] = k;
                a++;
            }
        }
    }
    checkCUDA(cudaMemcpy(aux_rmsd_ij_cu, aux_rmsd_ij.data(), nflex * npair * 2 * sizeof(int), cudaMemcpyHostToDevice));

    // set diagonal to 1 for aux_list_cluster_cu
    std::vector<int> aux_cluster_mat(nflex * dock_param.exhaustiveness, 1);
    checkCUDA(
        cudaMemcpy(aux_list_cluster_cu, aux_cluster_mat.data(), nflex * dock_param.exhaustiveness * sizeof(int),
            cudaMemcpyHostToDevice));


    //----- Wait for cudaMemcpy -----
    checkCUDA(cudaDeviceSynchronize()); // assure that memcpy is finished
    spdlog::info("Memory allocation on GPU is done.");
}


void DockTask::free_gpu(){
    //----- Free all GPU memory -----

    spdlog::info("Memory free on GPU...");
    
    if (flex_pose_list_manager){
        flex_pose_list_manager->free_all();
        delete flex_pose_list_manager;
        flex_pose_list_manager = nullptr;
        // flex_pose_list_cu is managed by flex_pose_list_manager
    }

    if (flex_topo_list_manager){
        flex_topo_list_manager->free_all();
        delete flex_topo_list_manager;
        flex_topo_list_manager = nullptr;
        // flex_topo_list_cu is managed by flex_topo_list_manager
    }

    checkCUDA(cudaFree(fix_mol_cu));
    checkCUDA(cudaFree(fix_mol_real_cu));

    if (flex_param_list_manager){
        flex_param_list_manager->free_all();
        delete flex_param_list_manager;
        flex_param_list_manager = nullptr;
        // flex_param_list_cu is managed by flex_param_list_manager
    }

    checkCUDA(cudaFree(fix_param_cu));
    checkCUDA(cudaFree(fix_param_int_cu));

    if (aux_poses_manager){
        aux_poses_manager->free_all();
        delete aux_poses_manager;
        aux_poses_manager = nullptr;
        // aux_poses_cu is managed by aux_poses_manager
    }

    if (aux_grads_manager){
        aux_grads_manager->free_all();
        delete aux_grads_manager;
        aux_grads_manager = nullptr;
        // aux_grads_cu is managed by aux_grads_manager
    }

    if (aux_hessians_manager){
        aux_hessians_manager->free_all();
        delete aux_hessians_manager;
        aux_hessians_manager = nullptr;
        // aux_hessians_cu is managed by aux_hessians_manager
    }

    if (aux_forces_manager){
        aux_forces_manager->free_all();
        delete aux_forces_manager;
        aux_forces_manager = nullptr;
        // aux_forces_cu is managed by aux_forces_manager
    }

    checkCUDA(cudaFree(aux_list_e_cu));
    checkCUDA(cudaFree(aux_list_cluster_cu));
    checkCUDA(cudaFree(aux_rmsd_ij_cu));
    checkCUDA(cudaFree(clustered_pose_inds_cu));

    spdlog::info("Memory free on GPU is done.");

    spdlog::info("Memory free on CPU...");
    checkCUDA(cudaFreeHost(flex_pose_list_res));
    checkCUDA(cudaFreeHost(flex_pose_list_real_res));
    spdlog::info("Memory free on CPU is done.");
}
