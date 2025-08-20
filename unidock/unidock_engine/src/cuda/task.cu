//
// Created by Congcong Liu on 24-11-13.
//

#include "common.cuh"
#include "task/DockTask.h"
#include "myutils/errors.h"


template <typename T>
void cp_to_device(T* device_ptr, const T* host_ptr, size_t count){
    checkCUDA(cudaMemcpy(device_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void cp_to_host(T* host_ptr, const T* device_ptr, size_t count){
    checkCUDA(cudaMemcpy(host_ptr, device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void alloc_device(T** device_ptr, size_t count){
    checkCUDA(cudaMalloc(device_ptr, count * sizeof(T)));
}

template <typename T>
void alloc_host(T** host_ptr, size_t count){
    checkCUDA(cudaMallocHost(host_ptr, count * sizeof(T)));
}


void alloc_cu_flex_pose_list(FlexPose** flex_pose_list_cu, Real** flex_pose_list_real_cu, std::vector<int>* list_i_real,
                             const UDFlexMolList& udflex_mols, int npose, int nflex, int exhaustiveness,
                             int* list_natom_flex, int n_atom_all_flex, int* list_ndihe, int n_dihe_all_flex){
    // GPU Cost: npose * sizeof(FlexPose) + exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex) * sizeof(Real)))

    alloc_device(flex_pose_list_cu, npose);
    alloc_device(flex_pose_list_real_cu, exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex));
    FlexPose* flex_pose_list; // for transferring to GPU
    alloc_host(&flex_pose_list, npose);
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

            flex_pose_list[i * exhaustiveness + j].coords = p_real_cu;
            list_i_real->push_back(list_i_real->back() + list_natom_flex[i] * 3);

            p_real_cu += list_natom_flex[i] * 3;
            flex_pose_list[i * exhaustiveness + j].dihedrals = p_real_cu;
            list_i_real->push_back(list_i_real->back() + list_ndihe[i]);

            // copy the coords and dihedrals to the GPU
            cp_to_device(flex_pose_list[i * exhaustiveness + j].coords, m.coords.data(), m.coords.size());
            cp_to_device(flex_pose_list[i * exhaustiveness + j].dihedrals, m.dihedrals.data(), m.dihedrals.size());

            // update the pointer
            p_real_cu += list_ndihe[i];
        }
    }
    cp_to_device(*flex_pose_list_cu, flex_pose_list, npose);
    checkCUDA(cudaFreeHost(flex_pose_list));
}


void DockTask::copy_all_to_cpu(){
    alloc_host(&flex_pose_list_res, nflex * dock_param.exhaustiveness);
    alloc_host(&flex_pose_list_real_res, dock_param.exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex));

    cp_to_host(flex_pose_list_res, flex_pose_list_cu, nflex * dock_param.exhaustiveness);
    cp_to_host(flex_pose_list_real_res, flex_pose_list_real_cu,
               dock_param.exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex));
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
    int size_bias_param_all_flex = 0;

    int list_natom_flex[nflex];
    int list_ndihe[nflex];

    int list_nrange[nflex]; // not used
    int list_ndim_trimat[nflex];
    int list_nrotated_atoms[nflex];

    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];

        size_intra_all_flex += m.intra_pairs.size();
        size_inter_all_flex += m.inter_pairs.size(); // all possible pairs

        list_natom_flex[i] = m.natom;
        list_ndihe[i] = m.dihedrals.size();
        n_atom_all_flex += m.natom;
        n_dihe_all_flex += m.dihedrals.size();

        int dim = 3 + 4 + m.dihedrals.size();
        n_dim_all_flex += dim;
        list_ndim_trimat[i] = dim * (dim + 1) / 2;
        n_dim_tri_mat_all_flex += list_ndim_trimat[i];
        list_nrotated_atoms[i] = 0;
        list_nrange[i] = 0;
        for (auto t : m.torsions){
            list_nrotated_atoms[i] += t.rotated_atoms.size();
            list_nrange[i] += t.range_list.size() / 2;
        }
        n_rotated_atoms_all_flex += list_nrotated_atoms[i];
        n_range_all_flex += list_nrange[i];

        size_bias_param_all_flex += m.biases.size() * 5;
    }

    // prepare pointers to CUDA continuous large memory
    Real* p_real_cu; //todo: Aiming to reduce page waste. However, maybe it is not necessary.
    int* p_int_cu;

    //----- flex_pose_list -----
    alloc_cu_flex_pose_list(&flex_pose_list_cu, &flex_pose_list_real_cu, &list_i_real, udflex_mols, npose, nflex,
                            dock_param.exhaustiveness,
                            list_natom_flex, n_atom_all_flex, list_ndihe, n_dihe_all_flex);


    //----- flex_topo_list -----
    // GPU cost: nflex * sizeof(FlexTopo) + (n_atom_all_flex + n_dihe_all_flex * 2 + n_dihe_all_flex * 2 +
    // n_dihe_all_flex * 2 + n_rotated_atoms_all_flex) * sizeof(int) +
    // (n_range_all_flex) * 2 * sizeof(Real)
    alloc_device(&flex_topo_list_cu, nflex);
    // vn_types, axis_atoms, range_inds, rotated_inds, rotated_atoms
    alloc_device(&flex_topo_list_int_cu,
                 n_atom_all_flex + n_dihe_all_flex * 2 + n_dihe_all_flex * 2 + n_dihe_all_flex * 2 +
                 n_rotated_atoms_all_flex);
    alloc_device(&flex_topo_list_real_cu, (n_range_all_flex) * 2); //range_list
    FlexTopo* flex_topo_list;
    alloc_host(&flex_topo_list, nflex);

    p_int_cu = flex_topo_list_int_cu;
    p_real_cu = flex_topo_list_real_cu;
    for (int i = 0; i < nflex; i++){
        int ind_range = 0, ind_rotated = 0;
        auto& m = udflex_mols[i];

        // set values
        flex_topo_list[i].natom = list_natom_flex[i];
        flex_topo_list[i].ntorsion = list_ndihe[i];

        // set pointer to GPU
        flex_topo_list[i].vn_types = p_int_cu;
        p_int_cu += list_natom_flex[i];
        flex_topo_list[i].axis_atoms = p_int_cu;
        p_int_cu += list_ndihe[i] * 2;
        flex_topo_list[i].range_inds = p_int_cu;
        p_int_cu += list_ndihe[i] * 2;
        flex_topo_list[i].rotated_inds = p_int_cu;
        p_int_cu += list_ndihe[i] * 2;
        flex_topo_list[i].rotated_atoms = p_int_cu;
        p_int_cu += list_nrotated_atoms[i];

        flex_topo_list[i].range_list = p_real_cu;

        // copy values to GPU
        cp_to_device(flex_topo_list[i].vn_types, m.vina_types.data(), m.natom);

        for (int j = 0; j < m.torsions.size(); j++){
            // todo: optimize memcpy by putting these to udmol
            cp_to_device(flex_topo_list[i].axis_atoms + j * 2, m.torsions[j].axis, 2);

            // range_inds end
            cp_to_device(flex_topo_list[i].range_inds + j * 2, &ind_range, 1);
            int n = m.torsions[j].range_list.size();
            // range list
            cp_to_device(flex_topo_list[i].range_list + ind_range, m.torsions[j].range_list.data(), n);
            // update pointer
            p_real_cu += n;
            // range_inds end
            ind_range += n;
            int nrange = n / 2;
            cp_to_device(flex_topo_list[i].range_inds + j * 2 + 1, &nrange, 1);


            // rotated_inds start
            cp_to_device(flex_topo_list[i].rotated_inds + j * 2, &ind_rotated, 1);
            // rotated_atoms
            cp_to_device(flex_topo_list[i].rotated_atoms + ind_rotated, m.torsions[j].rotated_atoms.data(),
                         m.torsions[j].rotated_atoms.size());

            // rotated_inds end
            ind_rotated += m.torsions[j].rotated_atoms.size() - 1;
            cp_to_device(flex_topo_list[i].rotated_inds + j * 2 + 1, &ind_rotated, 1);
            ind_rotated += 1;
        }
    }

    cp_to_device(flex_topo_list_cu, flex_topo_list, nflex);
    checkCUDA(cudaFreeHost(flex_topo_list));


    //----- fix_mol -----
    // GPU cost: sizeof(FixMol) + udfix_mol.coords.size() * sizeof(Real)
    alloc_device(&fix_mol_cu, 1);
    alloc_device(&fix_mol_real_cu, udfix_mol.coords.size());
    FixMol fix_mol;
    fix_mol.natom = udfix_mol.natom;
    fix_mol.coords = fix_mol_real_cu;
    cp_to_device(fix_mol.coords, udfix_mol.coords.data(), udfix_mol.coords.size());
    cp_to_device(fix_mol_cu, &fix_mol, 1);


    //----- flex_param_list_cu -----
    // GPU cost: nflex * sizeof(FlexParamVina) +
    // (size_intra_all_flex + size_inter_all_flex + n_atom_all_flex) * sizeof(int) +
    // (size_intra_all_flex + size_inter_all_flex) / 2 * sizeof(Real))
    alloc_device(&flex_param_list_cu, nflex);
    alloc_device(&flex_param_list_int_cu, (size_intra_all_flex + size_inter_all_flex + n_atom_all_flex + n_atom_all_flex * 2));
    alloc_device(&flex_param_list_real_cu, (size_intra_all_flex + size_inter_all_flex) / 2 + size_bias_param_all_flex);
    FlexParamVina* flex_param_list;
    alloc_host(&flex_param_list, nflex);

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

        flex_param_list[i].inds_bias = flex_param_list[i].atom_types + m.natom;

        flex_param_list[i].params_bias = flex_param_list[i].r1_plus_r2_inter + flex_param_list[i].npair_inter;

        // copy pairs_intra to cuda
        cp_to_device(flex_param_list[i].pairs_intra, m.intra_pairs.data(), m.intra_pairs.size());
        // copy pairs_inter to cuda
        cp_to_device(flex_param_list[i].pairs_inter, m.inter_pairs.data(), m.inter_pairs.size());
        // copy atom_types to cuda
        cp_to_device(flex_param_list[i].atom_types, m.vina_types.data(), m.natom);

        // copy inds_bias to cuda
        int inds_bias[m.natom * 2] = {0};
        std::vector<Real> params_bias;
        int last_i = -1;
        for (auto b : m.biases){
            if (b.i != last_i){
                inds_bias[b.i * 2] = params_bias.size();
            }
            params_bias.insert(params_bias.end(), b.param, b.param + 5);
            inds_bias[b.i * 2 + 1] = params_bias.size();
        }
        cp_to_device(flex_param_list[i].inds_bias, inds_bias, m.natom * 2);
        p_int_cu = flex_param_list[i].inds_bias + m.natom * 2;

        // copy r1_plus_r2_intra to cuda
        cp_to_device(flex_param_list[i].r1_plus_r2_intra, m.r1_plus_r2_intra.data(), m.r1_plus_r2_intra.size());
        // copy r1_plus_r2_inter to cuda
        cp_to_device(flex_param_list[i].r1_plus_r2_inter, m.r1_plus_r2_inter.data(), m.r1_plus_r2_inter.size());
        // copy bias_param to cuda
        cp_to_device(flex_param_list[i].params_bias, params_bias.data(), params_bias.size());

        p_real_cu = flex_param_list[i].params_bias + params_bias.size();
    }
    cp_to_device(flex_param_list_cu, flex_param_list, nflex);
    checkCUDA(cudaFreeHost(flex_param_list));


    //----- fix_param_cu -----
    // GPU cost: sizeof(FixParamVina) + udfix_mol.natom * sizeof(int)
    alloc_device(&fix_param_cu, 1);
    alloc_device(&fix_param_int_cu, udfix_mol.natom);
    FixParamVina fix_param;
    fix_param.atom_types = fix_param_int_cu;
    cp_to_device(fix_param.atom_types, udfix_mol.vina_types.data(), udfix_mol.natom);
    cp_to_device(fix_param_cu, &fix_param, 1);


    //----- aux_pose_cu -----
    // GPU cost: STRIDE_POSE * npose * sizeof(FlexPose)) +
    // TRIDE_POSE * dock_param.exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex)
    //        * sizeof(Real))
    //
    alloc_device(&aux_poses_cu, STRIDE_POSE * npose);
    alloc_device(&aux_poses_real_cu, STRIDE_POSE * dock_param.exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex));
    FlexPose* aux_poses;
    alloc_host(&aux_poses, STRIDE_POSE * npose);
    p_real_cu = aux_poses_real_cu;
    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            for (int k = 0; k < STRIDE_POSE; k++){
                aux_poses[i * STRIDE_POSE * dock_param.exhaustiveness + j * STRIDE_POSE + k].coords = p_real_cu;
                p_real_cu += list_natom_flex[i] * 3;
                aux_poses[i * STRIDE_POSE * dock_param.exhaustiveness + j * STRIDE_POSE + k].dihedrals = p_real_cu;
                p_real_cu += list_ndihe[i];
            }
        }
    }
    cp_to_device(aux_poses_cu, aux_poses, STRIDE_POSE * npose);
    checkCUDA(cudaFreeHost(aux_poses));


    //----- aux_grads_cu -----
    // GPU cost: STRIDE_G * npose * sizeof(FlexPoseGradient) +
    // STRIDE_G * dock_param.exhaustiveness * n_dihe_all_flex * sizeof(Real)
    alloc_device(&aux_grads_cu, STRIDE_G * npose);
    alloc_device(&aux_grads_real_cu, STRIDE_G * dock_param.exhaustiveness * n_dihe_all_flex);
    FlexPoseGradient* aux_grads;
    alloc_host(&aux_grads, STRIDE_G * npose);
    p_real_cu = aux_grads_real_cu;
    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            for (int k = 0; k < STRIDE_G; k++){
                aux_grads[i * STRIDE_G * dock_param.exhaustiveness + j * STRIDE_G + k].dihedrals_g = p_real_cu;
                p_real_cu += list_ndihe[i];
            }
        }
    }
    cp_to_device(aux_grads_cu, aux_grads, STRIDE_G * npose);
    checkCUDA(cudaFreeHost(aux_grads));


    //----- aux_hessians_cu -----
    // GPU cost:  npose * sizeof(FlexPoseHessian) +
    // dock_param.exhaustiveness * n_dim_tri_mat_all_flex * sizeof(Real)

    alloc_device(&aux_hessians_cu, npose);
    alloc_device(&aux_hessians_real_cu, dock_param.exhaustiveness * n_dim_tri_mat_all_flex);
    FlexPoseHessian* aux_hessians;
    alloc_host(&aux_hessians, npose);
    p_real_cu = aux_hessians_real_cu;
    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            aux_hessians[i * dock_param.exhaustiveness + j].matrix = p_real_cu;
            p_real_cu += list_ndim_trimat[i];
        }
    }
    cp_to_device(aux_hessians_cu, aux_hessians, npose);
    checkCUDA(cudaFreeHost(aux_hessians));


    //----- aux_forces_cu -----
    // GPU cost: npose * sizeof(FlexForce) +
    // dock_param.exhaustiveness * n_atom_all_flex * 3 * sizeof(Real)

    alloc_device(&aux_forces_cu, npose);
    alloc_device(&aux_forces_real_cu, dock_param.exhaustiveness * n_atom_all_flex * 3);
    FlexForce* aux_forces;
    alloc_host(&aux_forces, npose);
    p_real_cu = aux_forces_real_cu;
    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            aux_forces[i * dock_param.exhaustiveness + j].f = p_real_cu;
            p_real_cu += list_natom_flex[i] * 3;
        }
    }
    cp_to_device(aux_forces_cu, aux_forces, npose);
    checkCUDA(cudaFreeHost(aux_forces));


    //----- Clustering -----
    int npair = dock_param.exhaustiveness * (dock_param.exhaustiveness - 1) / 2; // tri-mat with diagonal
    alloc_device(&aux_list_e_cu, npose);
    alloc_device(&aux_list_cluster_cu, nflex * dock_param.exhaustiveness);
    alloc_device(&aux_rmsd_ij_cu, nflex * npair * 2);
    alloc_device(&clustered_pose_inds_cu, nflex * dock_param.exhaustiveness);

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
    cp_to_device(aux_rmsd_ij_cu, aux_rmsd_ij.data(), nflex * npair * 2);

    // set diagonal to 1 for aux_list_cluster_cu
    std::vector<int> aux_cluster_mat(nflex * dock_param.exhaustiveness, 1);
    cp_to_device(aux_list_cluster_cu, aux_cluster_mat.data(), nflex * dock_param.exhaustiveness);


    //----- Wait for cudaMemcpy -----
    checkCUDA(cudaDeviceSynchronize()); // assure that memcpy is finished
    spdlog::info("Memory allocation on GPU is done.");
}


void DockTask::free_gpu(){
    //----- Free all GPU memory -----

    spdlog::info("Memory free on GPU...");
    checkCUDA(cudaFree(flex_pose_list_cu));
    checkCUDA(cudaFree(flex_pose_list_real_cu));

    checkCUDA(cudaFree(flex_topo_list_cu));
    checkCUDA(cudaFree(flex_topo_list_int_cu));
    checkCUDA(cudaFree(flex_topo_list_real_cu));

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
