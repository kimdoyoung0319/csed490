#ifndef __PROJ1_DECOMP_H__
#define __PROJ1_DECOMP_H__

#include <vector>

#include "matrix.h"

struct DecompResult
{
    bool singular;
    std::vector<size_t> permutation;
    SquareMatrix lower;
    SquareMatrix upper;
};

DecompResult decompose(SquareMatrix target);

#endif