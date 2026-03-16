//
// Created by Congcong Liu on 24-11-21.
//

#ifndef CLUSTER_H
#define CLUSTER_H


#include "model/model.h"

void cluster_cu(int* out_clustered_pose_inds_cu, int* out_npose_clustered, std::vector<std::vector<int>>* clustered_pose_inds_list,
                const FlexPose* poses_cu, const FlexTopo* list_flex_topo,
                Real* aux_list_e_cu, Real* aux_rmsd_matrix_cu,
                int nflex, int exhaustiveness, Real rmsd_limit);

#endif //CLUSTER_H
