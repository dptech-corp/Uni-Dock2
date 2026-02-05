//
// Created by Congcong Liu on 24-9-23.
//

#ifndef JSON_H
#define JSON_H

#include <string>
#include "model/model.h"


void read_ud_from_json(const std::string &fp, const Box& box,
    UDFixMol& out_fix, UDFlexMolList& out_flex_list,
    std::vector<std::string>& out_fns_flex, bool use_tor_lib=true);

void read_ud_from_json_string(const std::string &ss, const Box& box,
    UDFixMol& out_fix, UDFlexMolList& out_flex_list,
    std::vector<std::string>& out_fns_flex, bool use_tor_lib=true);

void write_poses_to_json(std::string fp_json, const std::vector<std::string>& flex_names,
                         const std::vector<std::vector<int>>& filtered_pose_inds_list,
                         const FlexPose* flex_pose_list_res,
                         const Real* flex_pose_list_real_res,
                         const std::vector<size_t>& list_i_real);

#endif //JSON_H
