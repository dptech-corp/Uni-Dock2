//
// Created by Congcong Liu on 24-9-23.
//

#include "spdlog/spdlog.h"
#include <cmath>
#include <string>
#include <fstream>
#include <utility>
#include <rapidjson/document.h>

#include "json.h"
#include "rapidjson_parser.h"
#include "model/model.h"
#include "myutils/mymath.h"
#include "constants/constants.h"

namespace rj = rapidjson;

void read_ud_from_json(const std::string& fp, const Box& box, UDFixMol& out_fix, UDFlexMolList& out_flex_list,
                       std::vector<std::string>& out_fns_flex, bool use_tor_lib){

    std::ifstream ifs(fp);
    std::string json_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    read_ud_from_json_string(json_str, box,  out_fix, out_flex_list, out_fns_flex, use_tor_lib);

}


void read_ud_from_json_string(const std::string& json_str, const Box& box, UDFixMol& out_fix, UDFlexMolList& out_flex_list,
                       std::vector<std::string>& out_fns_flex, bool use_tor_lib) {

    rj::Document doc;
    doc.Parse(json_str.c_str());
    if (doc.HasParseError()){
        throw std::runtime_error("rapidjson: Failed to parse JSON string.");
    }

    spdlog::info("Json is successfully parsed");

    // Use RapidJsonParser to parse the data
    RapidJsonParser parser(doc);

    // Parse receptor
    Box box_protein;
    box_protein.x_lo = box.x_lo - VINA_CUTOFF;
    box_protein.x_hi = box.x_hi + VINA_CUTOFF;
    box_protein.y_lo = box.y_lo - VINA_CUTOFF;
    box_protein.y_hi = box.y_hi + VINA_CUTOFF;
    box_protein.z_lo = box.z_lo - VINA_CUTOFF;
    box_protein.z_hi = box.z_hi + VINA_CUTOFF;
    parser.parse_receptor_info(box_protein, out_fix);
    spdlog::info("Receptor has {:d} heavy atoms in box", out_fix.natom);

    // Parse ligands
    parser.parse_ligands_info(out_flex_list, out_fns_flex, use_tor_lib);
    spdlog::info("Flexible molecules count: {:d}", out_flex_list.size());
    if (out_flex_list.size() == 0){
        spdlog::error("No flexible molecules are found");
    }

    spdlog::debug("Json is Done.");
}

constexpr float MAX_SAFE_ENERGY = 1e6f;

auto safe_val = [](float v) -> float {
    if (std::isnan(v) || std::isinf(v)) return MAX_SAFE_ENERGY;
    if (v > MAX_SAFE_ENERGY) return MAX_SAFE_ENERGY;        
    if (v < -MAX_SAFE_ENERGY) return -MAX_SAFE_ENERGY;
    return v;
};


void collect_poses_to_map(
    PoseMap& out,
    const std::vector<std::string>& flex_names,
    const std::vector<std::vector<int>>& filtered_pose_inds_list,
    const FlexPose* flex_pose_list,
    const UDFlexMolList& udflex_mols,
    const std::vector<std::vector<std::vector<AtomEnergyDecomp>>>& decomp_list){

    for (int i = 0; i < static_cast<int>(flex_names.size()); i++){
        const auto& flex_name = flex_names[i];
        const int n_coords = udflex_mols[i].natom * 3;
        const int natom = udflex_mols[i].natom;
        const int n_dihe = static_cast<int>(udflex_mols[i].dihedrals.size());
        std::vector<PoseRecord> poses;

        int pose_idx = 0;
        for (auto& j: filtered_pose_inds_list[i]){
            PoseRecord pose;
            pose.energy = {
                safe_val(flex_pose_list[j].rot_vec[0]),
                safe_val(flex_pose_list[j].rot_vec[1]),
                safe_val(flex_pose_list[j].center[0]),
                safe_val(flex_pose_list[j].center[1]),
                safe_val(flex_pose_list[j].center[2]),
                safe_val(flex_pose_list[j].rot_vec[2]),
                safe_val(flex_pose_list[j].rot_vec[3]),
            };

            pose.coords.reserve(n_coords);
            for (int k = 0; k < n_coords; k++){
                pose.coords.push_back(safe_val(flex_pose_list[j].coords[k]));
            }

            pose.dihedrals.reserve(n_dihe);
            for (int k = 0; k < n_dihe; k++){
                pose.dihedrals.push_back(safe_val(rad_to_ang(flex_pose_list[j].dihedrals[k])));
            }

            if (i < static_cast<int>(decomp_list.size()) &&
                pose_idx < static_cast<int>(decomp_list[i].size())){
                const auto& atom_decomp = decomp_list[i][pose_idx];
                pose.decomp.reserve(natom);
                for (int a = 0; a < natom; a++){
                    pose.decomp.push_back({
                        safe_val(atom_decomp[a].gauss1),
                        safe_val(atom_decomp[a].gauss2),
                        safe_val(atom_decomp[a].repulsion),
                        safe_val(atom_decomp[a].hydrophobic),
                        safe_val(atom_decomp[a].hbond),
                    });
                }
            }

            poses.push_back(std::move(pose));
            pose_idx++;
        }

        out[flex_name] = std::move(poses);
    }
}
