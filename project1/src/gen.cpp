#include <random>

#include "gen.h"

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