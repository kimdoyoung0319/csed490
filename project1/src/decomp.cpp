#include <cmath>
#include <limits>
#include <omp.h>
#include <random>
#include <vector>

#include "decomp.h"
#include "matrix.h"

static void initialize(SquareMatrix &permutation, SquareMatrix &lower,
                       SquareMatrix &upper);
static size_t find_pivot(const SquareMatrix &matrix, size_t block);
static bool equal(const double a, const double b);

DecompResult decompose(SquareMatrix target)
{
    size_t size = target.size();

    SquareMatrix permutation(size), lower(size), upper(size);
    initialize(permutation, lower, upper);

    for (size_t block = 0; block < size; block++)
    {
        size_t pivot = find_pivot(target, block);

        if (equal(target(pivot, pivot), 0.0l))
        {
            return {true, permutation, lower, upper};
        }

        permutation.swap_rows(block, pivot);
        target.swap_rows(block, pivot);
        lower.swap_ranges(block, pivot, block);

        upper(block, block) = target(block, block);

#pragma omp parallel shared(target, lower, upper)
        {
#pragma omp for
            for (size_t i = block + 1; i < size; i++)
            {
                lower(i, block) = target(i, block) / upper(block, block);
                upper(block, i) = target(block, i);
            }

#pragma omp for
            for (size_t i = block + 1; i < size; i++)
            {
                for (size_t j = block + 1; j < size; j++)
                {
                    target(i, j) -= lower(i, block) * upper(block, j);
                }
            }
        }
    }

    return {false, permutation, lower, upper};
}

SquareMatrix generate(size_t size)
{
    std::random_device device;
    std::default_random_engine engine{device()};
    std::uniform_real_distribution distribution{0.0l, 10.0l};

    SquareMatrix result(size);

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            result(i, j) = distribution(engine);
        }
    }

    return result;
}

static void initialize(SquareMatrix &permutation, SquareMatrix &lower,
                       SquareMatrix &upper)
{
    size_t size = permutation.size();

#pragma omp prallel for shared(permutation)
    for (size_t i = 0; i < size; i++)
    {
        permutation(i, i) = 1.0l;
    }

#pragma omp prallel for shared(lower)
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
