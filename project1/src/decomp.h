#ifndef __PROJ1_DECOMP_H__
#define __PROJ1_DECOMP_H__

#include <vector>

#include "matrix.h"

struct DecompResult
{
    bool singular;
    SquareMatrix permutation;
    SquareMatrix lower;
    SquareMatrix upper;
};

SquareMatrix generate(size_t size);
DecompResult decompose(SquareMatrix target);

#endif