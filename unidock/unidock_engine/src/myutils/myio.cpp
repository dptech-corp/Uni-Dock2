//
// Created by Congcong Liu on 24-10-9.
//

#include "myio.h"

std::string gen_filepath(const std::string& fn, const std::string& dp_out) {
    std::string tmp = fn;
    if (dp_out == "") {
        return tmp;
    }
    return dp_out + separator() + tmp;
}



void my_info(const std::string& fn){
    std::cout << fn << std::endl << std::flush;
}
