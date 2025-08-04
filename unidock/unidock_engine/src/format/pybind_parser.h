#ifndef FORMAT_PYBIND_PARSER_H
#define FORMAT_PYBIND_PARSER_H

#include "parser_base.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

class PybindParser : public ParserBase {
public:
    PybindParser(py::list receptor_info, py::dict ligands_info);
    
    void parse_receptor_info(const Box& box_protein, UDFixMol& fix_mol) override;
    void parse_ligands_info(UDFlexMolList& flex_mol_list, std::vector<std::string>& fns_flex, bool use_tor_lib) override;

private:
    py::list receptor_info_;
    py::dict ligands_info_;
    
    void split_torsions_into_frags(const std::set<int>& root, const std::vector<UDTorsion>& torsions,
                                   std::vector<std::set<int>>& out_frags);
};
#endif // FORMAT_PYBIND_PARSER_H