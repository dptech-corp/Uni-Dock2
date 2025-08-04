#include "pybind_parser.h"
#include "myutils/mymath.h"
#include "myutils/common.h"
#include "constants/constants.h"
#include <spdlog/spdlog.h>

#include <set>
#include <numeric>
#include <algorithm>

PybindParser::PybindParser(py::list receptor_info, py::dict ligands_info) 
    : receptor_info_(receptor_info), ligands_info_(ligands_info) {}

void PybindParser::parse_receptor_info(const Box& box_protein, UDFixMol& fix_mol) {
    fix_mol.coords.clear();
    fix_mol.vina_types.clear();
    fix_mol.ff_types.clear();
    fix_mol.charges.clear();
    
    for (const auto& atom_info : receptor_info_) {
        py::list atom_list = atom_info.cast<py::list>();
        if (atom_list[3].cast<int>() == VN_TYPE_H) { // remove Hydrogens
            continue;
        }

        Real x = atom_list[0].cast<Real>();
        Real y = atom_list[1].cast<Real>();
        Real z = atom_list[2].cast<Real>();
        if (box_protein.is_inside(x, y, z)) { // for acceleration
            fix_mol.coords.push_back(x);
            fix_mol.coords.push_back(y);
            fix_mol.coords.push_back(z);
            fix_mol.vina_types.push_back(atom_list[3].cast<int>());
            fix_mol.ff_types.push_back(atom_list[4].cast<int>());
            fix_mol.charges.push_back(atom_list[5].cast<Real>());
        }
    }
    fix_mol.natom = fix_mol.charges.size();
}

void PybindParser::parse_ligands_info(UDFlexMolList& flex_mol_list, std::vector<std::string>& fns_flex, bool use_tor_lib) {
    std::set<std::string> exclude_keys = {"score", "receptor"};
    
    for (const auto &[k, v] : ligands_info_) {
        std::string key = k.cast<std::string>();
        if (exclude_keys.find(key) != exclude_keys.end()) {
            continue;
        }

        fns_flex.push_back(key);

        UDFlexMol flex_mol;
        Real coords_sum[3] = {0};
        
        py::dict ligand_dict = v.cast<py::dict>();
        
        // Atoms
        py::list atom_info = ligand_dict["atoms"].cast<py::list>();
        for (int ia = 0; ia < atom_info.size(); ia++) {
            py::list atom_line = atom_info[ia].cast<py::list>();
            Real x = atom_line[0].cast<Real>();
            Real y = atom_line[1].cast<Real>();
            Real z = atom_line[2].cast<Real>();
            flex_mol.coords.push_back(x);
            flex_mol.coords.push_back(y);
            flex_mol.coords.push_back(z);
            coords_sum[0] += x;
            coords_sum[1] += y;
            coords_sum[2] += z;

            flex_mol.vina_types.push_back(atom_line[3].cast<int>());
            flex_mol.ff_types.push_back(atom_line[4].cast<int>());
            flex_mol.charges.push_back(atom_line[5].cast<Real>());

            py::list pairs_12_13 = atom_line[6].cast<py::list>();
            for (const auto& a : pairs_12_13) {
                auto ib = a.cast<int>();
                if (ia == ib) {
                    spdlog::warn("Remove wrong pair 1-2 & 1-3 for Self: flex {} atom {} - {}", key, ia, ib);
                } else {
                    flex_mol.pairs_1213.insert(order_pair(ia, ib));
                }
            }
            
            py::list pairs_14 = atom_line[7].cast<py::list>();
            for (const auto& a : pairs_14) {
                auto ib = a.cast<int>();
                if (ia == ib) {
                    spdlog::warn("Remove wrong pair 1-4 for Self: flex {} atom {} - {}", key, ia, ib);
                } else {
                    auto p = order_pair(ia, ib);
                    if (flex_mol.pairs_1213.find(p) != flex_mol.pairs_1213.end()) {
                        spdlog::warn("Remove wrong pair 1-4 for Already in 1-2&1-3: flex {} atom {} - {}", key, ia, ib);
                    } else {
                        flex_mol.pairs_14.insert(p);
                    }
                }
            }
        }

        std::set<int> root_atoms;
        py::list root_info = ligand_dict["root_atoms"].cast<py::list>();
        for (const auto& a : root_info) {
            root_atoms.insert(a.cast<int>());
        }

        // Torsions
        py::list torsions_info = ligand_dict["torsions"].cast<py::list>();
        for (const auto& torsion_info : torsions_info) {
            py::list tor_info = torsion_info.cast<py::list>();
            UDTorsion torsion;

            py::list torsion_atoms = tor_info[0].cast<py::list>();
            for (int i = 0; i < 4; i++) {
                torsion.atoms[i] = torsion_atoms[i].cast<int>();
            }

            // axis is the two middle atoms
            torsion.axis[0] = torsion.atoms[1];
            torsion.axis[1] = torsion.atoms[2];

            // dihedral value
            flex_mol.dihedrals.push_back(ang_to_rad(tor_info[1].cast<Real>()));

            // range list
            if (use_tor_lib) {
                py::list range_list = tor_info[2].cast<py::list>();
                for (const auto& r : range_list) {
                    py::list range_pair = r.cast<py::list>();
                    Real rad_lo = ang_to_rad(range_pair[0].cast<Real>());
                    Real rad_hi = ang_to_rad(range_pair[1].cast<Real>());
                    // For range crossing "180/-180", split it into to ranges
                    if (rad_lo < rad_hi) {
                        torsion.range_list.push_back(rad_lo);
                        torsion.range_list.push_back(rad_hi);
                    } else if (rad_lo > 0 && rad_hi < 0) {
                        torsion.range_list.push_back(rad_lo);
                        torsion.range_list.push_back(PI);
                        torsion.range_list.push_back(-PI);
                        torsion.range_list.push_back(rad_hi);
                    } else {
                        spdlog::critical("Input json has wrong range list, rad_lo > rad_hi: {}, {}",
                                         range_pair[0].cast<Real>(), range_pair[1].cast<Real>());
                    }
                }
            } else {
                torsion.range_list.push_back(-PI);
                torsion.range_list.push_back(PI);
            }

            // rotated atoms
            py::list rotated_atoms = tor_info[3].cast<py::list>();
            for (const auto& a : rotated_atoms) {
                torsion.rotated_atoms.push_back(a.cast<int>());
            }
            
            // gaff2 parameters, may be multiple groups
            py::list gaff2_params = tor_info[4].cast<py::list>();
            for (const auto& gaff2 : gaff2_params) {
                py::list gaff2_group = gaff2.cast<py::list>();
                torsion.param_gaff2.push_back(gaff2_group[0].cast<Real>());
                torsion.param_gaff2.push_back(gaff2_group[1].cast<Real>());
                torsion.param_gaff2.push_back(gaff2_group[2].cast<Real>());
                torsion.param_gaff2.push_back(gaff2_group[3].cast<Real>());
            }
            
            flex_mol.torsions.push_back(torsion);
        }

        std::vector<std::set<int>> frags;
        split_torsions_into_frags(root_atoms, flex_mol.torsions, frags);
        
        // Compute necessary properties
        flex_mol.name = key;
        flex_mol.natom = flex_mol.charges.size();
        flex_mol.center[0] = coords_sum[0] / flex_mol.natom;
        flex_mol.center[1] = coords_sum[1] / flex_mol.natom;
        flex_mol.center[2] = coords_sum[2] / flex_mol.natom;

        // intra pairs
        for (int i = 0; i < flex_mol.natom; i++) {
            for (int j = i + 1; j < flex_mol.natom; j++) {
                // exclude 1-2, 1-3, 1-4 pairs
                if ((flex_mol.pairs_1213.find(order_pair(i, j)) == flex_mol.pairs_1213.end()) &&
                    (flex_mol.pairs_14.find(order_pair(i, j)) == flex_mol.pairs_14.end()) &&
                    (!checkInAnySameSet(frags, i, j))) {
                    flex_mol.intra_pairs.push_back(i);
                    flex_mol.intra_pairs.push_back(j);
                }
            }
        }
        
        // Note: inter pairs will need to be handled separately when receptor is available
        
        flex_mol_list.push_back(flex_mol);
    }
}

void PybindParser::split_torsions_into_frags(const std::set<int>& root, const std::vector<UDTorsion>& torsions,
                                             std::vector<std::set<int>>& out_frags) {
    // The first frag is the root atoms
    std::set<int> frag0;
    for (int i : root) {
        frag0.insert(i);
    }
    
    // use index list
    std::vector<int> range(torsions.size());
    std::iota(range.begin(), range.end(), 0);
    
    // Step 1: Identify big torsions and add them as fragments
    std::vector<int> range_tmp = range;
    for (int itor : range_tmp) {
        auto& torsion = torsions[itor];
        // Check if either axis atom is in the root set
        if (root.find(torsion.axis[0]) != root.end() || root.find(torsion.axis[1]) != root.end()) {
            // remove this itor from the list
            range.erase(std::remove(range.begin(), range.end(), itor), range.end());
            
            // also add axis to the root frag
            frag0.insert(torsion.axis[0]);
            frag0.insert(torsion.axis[1]);
            
            // Create a new fragment for this big torsion
            std::set<int> frag_tmp;
            // add to this new frag
            for (int i : torsion.rotated_atoms) {
                frag_tmp.insert(i);
            }
            // also add axis to this frag
            frag_tmp.insert(torsion.axis[0]);
            frag_tmp.insert(torsion.axis[1]);
            out_frags.push_back(frag_tmp);
        }
    }
    out_frags.push_back(frag0);
    
    // Step 2: Split existing fragments using other torsions
    for (int itor : range) {
        auto& torsion = torsions[itor];
        
        // Check if the torsion's axis atoms belong to an existing fragment
        for (auto& frag : out_frags) {
            if ((frag.find(torsion.axis[0]) != frag.end()) && (frag.find(torsion.axis[1]) != frag.end())) {
                // Split the fragment into two parts
                std::set<int> frag1, frag2;
                std::set<int> tor_set(torsion.rotated_atoms.begin(), torsion.rotated_atoms.end());
                std::set_intersection(frag.begin(), frag.end(),
                                      tor_set.begin(), tor_set.end(),
                                      std::inserter(frag1, frag1.begin()));
                std::set_difference(frag.begin(), frag.end(),
                                    tor_set.begin(), tor_set.end(),
                                    std::inserter(frag2, frag2.begin()));
                
                // also add the axis
                frag1.insert(torsion.axis[0]);
                frag1.insert(torsion.axis[1]);
                frag2.insert(torsion.axis[0]);
                frag2.insert(torsion.axis[1]);
                
                // Replace the original fragment with the two new fragments
                frag = frag1;
                out_frags.push_back(frag2);
                break; // Move to the next torsion
            }
        }
    }
}