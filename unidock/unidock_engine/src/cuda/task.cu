//
// Created by Congcong Liu on 24-11-13.
//

#include <cstring>
#include <iterator>
#include <algorithm>
#include "common.cuh"
#include "task/DockTask.h"
#include "myutils/errors.h"
#include "cuda/struct_array_manager.cuh"


void alloc_flex_topo_list(StructArrayManager<FlexTopo>*& flex_topo_list_manager, FlexTopo*& flex_topo_list_cu,
                          const UDFlexMolList& udflex_mols,
                          const std::vector<int>& list_n_atom_flex, const std::vector<int>& list_n_dihe,
                          const std::vector<int>& list_n_range, const std::vector<int>& list_n_rotated_atoms){

    int nflex = udflex_mols.size();
    flex_topo_list_manager = new StructArrayManager<FlexTopo>(nflex);

    std::vector<int> list_n_dihe_x2(nflex);
    std::transform(list_n_dihe.begin(), list_n_dihe.end(), list_n_dihe_x2.begin(), [](int v){ return v * 2; });
    std::vector<int> list_range_list_size(nflex);
    std::transform(list_n_range.begin(), list_n_range.end(), list_range_list_size.begin(), [](int v){ return v * 2; });

    flex_topo_list_manager->add_ptr_field<int*>({&FlexTopo::vn_types, sizeof(int), list_n_atom_flex});
    flex_topo_list_manager->add_ptr_field<int*>({&FlexTopo::axis_atoms, sizeof(int), list_n_dihe_x2});
    flex_topo_list_manager->add_ptr_field<int*>({&FlexTopo::range_inds, sizeof(int), list_n_dihe_x2});
    flex_topo_list_manager->add_ptr_field<int*>({&FlexTopo::rotated_inds, sizeof(int), list_n_dihe_x2});
    flex_topo_list_manager->add_ptr_field<int*>({&FlexTopo::rotated_atoms, sizeof(int), list_n_rotated_atoms});
    flex_topo_list_manager->add_ptr_field<Real*>({&FlexTopo::range_list, sizeof(Real), list_range_list_size});

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
    checkCUDA(cudaMallocHost(&flex_pose_list_res, nflex * dock_param.exhaustiveness * sizeof(FlexPose)));
    checkCUDA(
        cudaMallocHost(&flex_pose_list_real_res, dock_param.exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex) *
            sizeof(Real)));

    checkCUDA(cudaMemcpy(flex_pose_list_res, flex_pose_list_cu, nflex * dock_param.exhaustiveness * sizeof(FlexPose),
        cudaMemcpyDeviceToHost));
    checkCUDA(
        cudaMemcpy(flex_pose_list_real_res, flex_pose_list_real_cu, dock_param.exhaustiveness * (n_atom_all_flex * 3 +
                n_dihe_all_flex) * sizeof(Real),
            cudaMemcpyDeviceToHost));
}


void alloc_cu_flex_pose_list(FlexPose** flex_pose_list_cu, Real** flex_pose_list_real_cu, std::vector<int>* list_i_real,
                             const UDFlexMolList& udflex_mols, int npose, int nflex, int exhaustiveness,
                             int* list_natom_flex, int n_atom_all_flex, int* list_ndihe, int n_dihe_all_flex){
    // GPU Cost: npose * sizeof(FlexPose) + exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex) * sizeof(Real)))

    checkCUDA(cudaMalloc(flex_pose_list_cu, npose * sizeof(FlexPose)));
    checkCUDA(cudaMalloc(flex_pose_list_real_cu,
        exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex) * sizeof(Real)));
    FlexPose* flex_pose_list; // for transferring to GPU
    checkCUDA(cudaMallocHost(&flex_pose_list, npose * sizeof(FlexPose)));
    // set the value of flex_pose_list
    Real* p_real_cu = *flex_pose_list_real_cu;

    list_i_real->push_back(0);
    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];
        for (int j = 0; j < exhaustiveness; j++){
            // I prefer to randomize pose on GPU. So here is just a copy of initial pose
            flex_pose_list[i * exhaustiveness + j].center[0] = m.center[0];
            flex_pose_list[i * exhaustiveness + j].center[1] = m.center[1];
            flex_pose_list[i * exhaustiveness + j].center[2] = m.center[2];
            flex_pose_list[i * exhaustiveness + j].rot_vec[0] = 0;
            flex_pose_list[i * exhaustiveness + j].rot_vec[1] = 0;
            flex_pose_list[i * exhaustiveness + j].rot_vec[2] = 0;
            flex_pose_list[i * exhaustiveness + j].energy = 999;

            flex_pose_list[i * exhaustiveness + j].coords = p_real_cu;
            list_i_real->push_back(list_i_real->back() + list_natom_flex[i] * 3);

            p_real_cu += list_natom_flex[i] * 3;
            flex_pose_list[i * exhaustiveness + j].dihedrals = p_real_cu;
            list_i_real->push_back(list_i_real->back() + list_ndihe[i]);

            // copy the coords and dihedrals to the GPU
            checkCUDA(
                cudaMemcpy(flex_pose_list[i * exhaustiveness + j].coords, m.coords.data(), m.coords.size() * sizeof(Real
                    ),
                    cudaMemcpyHostToDevice));
            checkCUDA(
                cudaMemcpy(flex_pose_list[i * exhaustiveness + j].dihedrals, m.dihedrals.data(), m.dihedrals.size() *
                    sizeof(Real),
                    cudaMemcpyHostToDevice));

            // update the pointer
            p_real_cu += list_ndihe[i];
        }
    }
    checkCUDA(cudaMemcpy(*flex_pose_list_cu, flex_pose_list, npose * sizeof(FlexPose),
        cudaMemcpyHostToDevice));
    checkCUDA(cudaFreeHost(flex_pose_list));
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
    // prepare pointers to CUDA continuous large memory
    Real* p_real_cu;
    int* p_int_cu;

    //----- flex_pose_list -----
    alloc_cu_flex_pose_list(&flex_pose_list_cu, &flex_pose_list_real_cu, &list_i_real, udflex_mols, npose, nflex,
                            dock_param.exhaustiveness,
                            list_n_atom_flex.data(), n_atom_all_flex, list_n_dihe.data(), n_dihe_all_flex);


    //----- flex_topo_list -----
    alloc_flex_topo_list(flex_topo_list_manager, flex_topo_list_cu,
                         udflex_mols, list_n_atom_flex, list_n_dihe, list_n_range, list_n_rotated_atoms);


    //----- fix_mol -----
    // GPU cost: sizeof(FixMol) + udfix_mol.coords.size() * sizeof(Real)
    alloc_fix_mol(fix_mol_cu, fix_mol_real_cu, udfix_mol);


    //----- flex_param_list_cu -----
    // GPU cost: nflex * sizeof(FlexParamVina) +
    // (size_intra_all_flex + size_inter_all_flex + n_atom_all_flex) * sizeof(int) +
    // (size_intra_all_flex + size_inter_all_flex) / 2 * sizeof(Real))
    checkCUDA(cudaMalloc(&flex_param_list_cu, nflex * sizeof(FlexParamVina)));
    checkCUDA(
        cudaMalloc(&flex_param_list_int_cu, (size_intra_all_flex + size_inter_all_flex + n_atom_all_flex) * sizeof(int)
        ));
    checkCUDA(cudaMalloc(&flex_param_list_real_cu, (size_intra_all_flex + size_inter_all_flex) / 2 * sizeof(Real)));
    FlexParamVina* flex_param_list;
    checkCUDA(cudaMallocHost(&flex_param_list, nflex * sizeof(FlexParamVina)));

    p_int_cu = flex_param_list_int_cu;
    p_real_cu = flex_param_list_real_cu;
    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];
        flex_param_list[i].npair_intra = m.intra_pairs.size() / 2;
        flex_param_list[i].pairs_intra = p_int_cu;
        flex_param_list[i].r1_plus_r2_intra = p_real_cu;

        flex_param_list[i].npair_inter = m.inter_pairs.size() / 2;
        flex_param_list[i].pairs_inter = flex_param_list[i].pairs_intra + flex_param_list[i].npair_intra * 2;
        flex_param_list[i].r1_plus_r2_inter = p_real_cu + flex_param_list[i].npair_intra;

        flex_param_list[i].atom_types = flex_param_list[i].pairs_inter + flex_param_list[i].npair_inter * 2;

        // copy pairs_intra to cuda
        checkCUDA(
            cudaMemcpy(flex_param_list[i].pairs_intra, m.intra_pairs.data(), m.intra_pairs.size() * sizeof(int),
                cudaMemcpyHostToDevice));
        // copy pairs_inter to cuda
        checkCUDA(
            cudaMemcpy(flex_param_list[i].pairs_inter, m.inter_pairs.data(), m.inter_pairs.size() * sizeof(int),
                cudaMemcpyHostToDevice));
        // copy atom_types to cuda
        checkCUDA(
            cudaMemcpy(flex_param_list[i].atom_types, m.vina_types.data(), m.natom * sizeof(int), cudaMemcpyHostToDevice
            ));
        p_int_cu = flex_param_list[i].atom_types + m.natom;
        // copy r1_plus_r2_intra to cuda
        checkCUDA(
            cudaMemcpy(flex_param_list[i].r1_plus_r2_intra, m.r1_plus_r2_intra.data(), m.r1_plus_r2_intra.size() *
                sizeof(Real), cudaMemcpyHostToDevice));
        // copy r1_plus_r2_inter to cuda
        checkCUDA(
            cudaMemcpy(flex_param_list[i].r1_plus_r2_inter, m.r1_plus_r2_inter.data(), m.r1_plus_r2_inter.size() *
                sizeof(Real), cudaMemcpyHostToDevice));
        p_real_cu = flex_param_list[i].r1_plus_r2_inter + m.r1_plus_r2_inter.size();
    }
    checkCUDA(cudaMemcpy(flex_param_list_cu, flex_param_list, nflex * sizeof(FlexParamVina), cudaMemcpyHostToDevice));
    checkCUDA(cudaFreeHost(flex_param_list));


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
    // GPU cost: STRIDE_POSE * npose * sizeof(FlexPose)) +
    // TRIDE_POSE * dock_param.exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex)
    //        * sizeof(Real))
    //
    checkCUDA(cudaMalloc(&aux_poses_cu, STRIDE_POSE * npose * sizeof(FlexPose)));
    checkCUDA(
        cudaMalloc(&aux_poses_real_cu, STRIDE_POSE * dock_param.exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex)
            * sizeof(Real)));
    FlexPose* aux_poses;
    checkCUDA(cudaMallocHost(&aux_poses, STRIDE_POSE * npose * sizeof(FlexPose)));
    p_real_cu = aux_poses_real_cu;
    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            for (int k = 0; k < STRIDE_POSE; k++){
                aux_poses[i * STRIDE_POSE * dock_param.exhaustiveness + j * STRIDE_POSE + k].coords = p_real_cu;
                p_real_cu += list_n_atom_flex[i] * 3;
                aux_poses[i * STRIDE_POSE * dock_param.exhaustiveness + j * STRIDE_POSE + k].dihedrals = p_real_cu;
                p_real_cu += list_n_dihe[i];
            }
        }
    }
    checkCUDA(cudaMemcpy(aux_poses_cu, aux_poses, STRIDE_POSE * npose * sizeof(FlexPose), cudaMemcpyHostToDevice));
    checkCUDA(cudaFreeHost(aux_poses));


    //----- aux_grads_cu -----
    // GPU cost: STRIDE_G * npose * sizeof(FlexPoseGradient) +
    // STRIDE_G * dock_param.exhaustiveness * n_dihe_all_flex * sizeof(Real)
    checkCUDA(cudaMalloc(&aux_grads_cu, STRIDE_G * npose * sizeof(FlexPoseGradient)));
    checkCUDA(cudaMalloc(&aux_grads_real_cu, STRIDE_G * dock_param.exhaustiveness * n_dihe_all_flex * sizeof(Real)));
    FlexPoseGradient* aux_grads;
    checkCUDA(cudaMallocHost(&aux_grads, STRIDE_G * npose * sizeof(FlexPoseGradient)));
    p_real_cu = aux_grads_real_cu;
    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            for (int k = 0; k < STRIDE_G; k++){
                aux_grads[i * STRIDE_G * dock_param.exhaustiveness + j * STRIDE_G + k].dihedrals_g = p_real_cu;
                p_real_cu += list_n_dihe[i];
            }
        }
    }
    checkCUDA(cudaMemcpy(aux_grads_cu, aux_grads, STRIDE_G * npose * sizeof(FlexPoseGradient), cudaMemcpyHostToDevice));
    checkCUDA(cudaFreeHost(aux_grads));


    //----- aux_hessians_cu -----
    // GPU cost:  npose * sizeof(FlexPoseHessian) +
    // dock_param.exhaustiveness * n_dim_tri_mat_all_flex * sizeof(Real)

    checkCUDA(cudaMalloc(&aux_hessians_cu, npose * sizeof(FlexPoseHessian)));
    checkCUDA(cudaMalloc(&aux_hessians_real_cu, dock_param.exhaustiveness * n_dim_tri_mat_all_flex * sizeof(Real)));
    FlexPoseHessian* aux_hessians;
    checkCUDA(cudaMallocHost(&aux_hessians, npose * sizeof(FlexPoseHessian)));
    p_real_cu = aux_hessians_real_cu;
    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            aux_hessians[i * dock_param.exhaustiveness + j].matrix = p_real_cu;
            p_real_cu += list_ndim_trimat[i];
        }
    }
    checkCUDA(cudaMemcpy(aux_hessians_cu, aux_hessians, npose * sizeof(FlexPoseHessian), cudaMemcpyHostToDevice));
    checkCUDA(cudaFreeHost(aux_hessians));


    //----- aux_forces_cu -----
    // GPU cost: npose * sizeof(FlexForce) +
    // dock_param.exhaustiveness * n_atom_all_flex * 3 * sizeof(Real)

    checkCUDA(cudaMalloc(&aux_forces_cu, npose * sizeof(FlexForce)));
    checkCUDA(cudaMalloc(&aux_forces_real_cu, dock_param.exhaustiveness * n_atom_all_flex * 3 * sizeof(Real)));
    FlexForce* aux_forces;
    checkCUDA(cudaMallocHost(&aux_forces, npose * sizeof(FlexForce)));
    p_real_cu = aux_forces_real_cu;
    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            aux_forces[i * dock_param.exhaustiveness + j].f = p_real_cu;
            p_real_cu += list_n_atom_flex[i] * 3;
        }
    }
    checkCUDA(cudaMemcpy(aux_forces_cu, aux_forces, npose * sizeof(FlexForce), cudaMemcpyHostToDevice));
    checkCUDA(cudaFreeHost(aux_forces));


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
    checkCUDA(cudaFree(flex_pose_list_cu));
    checkCUDA(cudaFree(flex_pose_list_real_cu));

    if (flex_topo_list_manager){
        flex_topo_list_manager->free_all();
        delete flex_topo_list_manager;
        flex_topo_list_manager = nullptr;
        // flex_topo_list_cu is managed by flex_topo_list_manager
    }

    checkCUDA(cudaFree(fix_mol_cu));
    checkCUDA(cudaFree(fix_mol_real_cu));

    checkCUDA(cudaFree(flex_param_list_cu));
    checkCUDA(cudaFree(flex_param_list_int_cu));
    checkCUDA(cudaFree(flex_param_list_real_cu));

    checkCUDA(cudaFree(fix_param_cu));
    checkCUDA(cudaFree(fix_param_int_cu));

    checkCUDA(cudaFree(aux_poses_cu));
    checkCUDA(cudaFree(aux_poses_real_cu));

    checkCUDA(cudaFree(aux_grads_cu));
    checkCUDA(cudaFree(aux_grads_real_cu));

    checkCUDA(cudaFree(aux_hessians_cu));
    checkCUDA(cudaFree(aux_hessians_real_cu));

    checkCUDA(cudaFree(aux_forces_cu));
    checkCUDA(cudaFree(aux_forces_real_cu));

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
