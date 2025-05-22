//
// Created by Congcong Liu on 24-9-24.
//

#ifndef MYIO_H
#define MYIO_H

#include <string>
#include <iostream>

inline char separator() {
    // Source:
    // https://stackoverflow.com/questions/12971499/how-to-get-the-file-separator-symbol-in-standard-c-c-or
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}


std::string gen_filepath(const std::string& fn, const std::string& dp_out);

void my_info(const std::string& fn);


#endif //MYIO_H
