#include "decomp.h"
#include "matrix.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

static void initialize(std::vector<size_t> &permutation, SquareMatrix &lower,
                       SquareMatrix &upper);
static size_t find_pivot(const SquareMatrix &matrix, size_t block);
static bool equal(const double a, const double b);

DecompResult decompose(SquareMatrix target)
{
    size_t size = target.size();

    std::vector<size_t> permutation(size);
    SquareMatrix lower(size);
    SquareMatrix upper(size);

    initialize(permutation, lower, upper);

    for (size_t block = 0; block < size; block++)
    {
        size_t pivot = find_pivot(target, block);

        if (equal(target(pivot, pivot), 0.0l))
        {
            return {true, permutation, lower, upper};
        }

        std::swap(permutation[block], permutation[pivot]);
        target.swap_rows(block, pivot);
        lower.swap_ranges(block, pivot, block);

        upper(block, block) = target(block, block);

        for (size_t i = block + 1; i < size; i++)
        {
            lower(i, block) = target(i, block) / upper(block, block);
            upper(block, i) = target(block, i);
        }

        for (size_t i = block + 1; i < size; i++)
        {
            for (size_t j = block + 1; j < size; j++)
            {
                target(i, j) -= lower(i, block) * upper(block, j);
            }
        }
    }

    return {false, permutation, lower, upper};
}

static void initialize(std::vector<size_t> &permutation, SquareMatrix &lower,
                       SquareMatrix &upper)
{
    size_t size = permutation.size();

    for (size_t i = 0; i < size; i++)
    {
        permutation[i] = i;
    }

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            lower(i, j) = 0.0l;
        }

        lower(i, i) = 1.0l;
    }
}

static size_t find_pivot(const SquareMatrix &matrix, size_t block)
{
    size_t size = matrix.size();
    size_t max_row = block;

    for (size_t row = block; row < size; row++)
    {
        max_row = (matrix(max_row, block) < matrix(row, block)) ? row : max_row;
    }

    return max_row;
}

static bool equal(const double a, const double b)
{
    const double epsilon = std::numeric_limits<double>::epsilon();
    return fabs(a - b) <= ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}
