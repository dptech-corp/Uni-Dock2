#ifndef FORMAT_PARSER_BASE_H
#define FORMAT_PARSER_BASE_H

#include <vector>
#include <string>
#include <set>

#include "model/model.h"
#include "constants/constants.h"
#include "myutils/common.h"

class ParserBase {
public:
    virtual ~ParserBase() = default;
    
    virtual void parse_receptor_info(const Box& box_protein, UDFixMol& fix_mol) = 0;
    virtual void parse_ligands_info(UDFlexMolList& flex_mol_list, std::vector<std::string>& fns_flex, bool use_tor_lib) = 0;
    
protected:
    SCOPE_INLINE std::pair<int, int> order_pair(int a, int b) {
        return std::make_pair(std::min(a, b), std::max(a, b));
    }
    
    bool checkInAnySameSet(const std::vector<std::set<int>>& frags, int v1, int v2) {
        for (const auto& frag : frags) {
            if (frag.find(v1) != frag.end() && frag.find(v2) != frag.end()) {
                return true;
            }
        }
        return false;
    }
};
#endif // FORMAT_PARSER_BASE_H