//
// Created by Congcong Liu on 24-9-23.
//

#ifndef JSON_H
#define JSON_H

#include <map>
#include <string>
#include <vector>
#include "model/model.h"
#include "score/score.h"

struct PoseRecord {
    std::vector<Real> energy;
    std::vector<Real> coords;
    std::vector<Real> dihedrals;
    std::vector<std::vector<Real>> decomp;
};

using PoseMap = std::map<std::string, std::vector<PoseRecord>>;

void read_ud_from_json(const std::string &fp, const Box& box,
    UDFixMol& out_fix, UDFlexMolList& out_flex_list,
    std::vector<std::string>& out_fns_flex, bool use_tor_lib=true);

void read_ud_from_json_string(const std::string &ss, const Box& box,
    UDFixMol& out_fix, UDFlexMolList& out_flex_list,
    std::vector<std::string>& out_fns_flex, bool use_tor_lib=true);

void collect_poses_to_map(
    PoseMap& out,
    const std::vector<std::string>& flex_names,
    const std::vector<std::vector<int>>& filtered_pose_inds_list,
    const FlexPose* flex_pose_list,
    const UDFlexMolList& udflex_mols,
    const std::vector<std::vector<std::vector<AtomEnergyDecomp>>>& decomp_list);

#endif //JSON_H
