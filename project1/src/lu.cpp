#include <chrono>
#include <iostream>
#include <omp.h>

#include "decomp.h"
#include "matrix.h"

void usage(const char *name)
{
    std::cout << "Usage: " << name << " matrix-size nworkers [options]\n"
              << "Options:\n"
              << "-v  Prints the generated matrix and decomposed matrices.\n"
              << "    Do not recommend for huge matrices since it will mess up "
                 "your shell.\n"
              << std::endl;

    exit(-1);
}

int main(int argc, char **argv)
{
    bool verbose = false;
    const char *name = argv[0];

    if (argc < 3)
    {
        usage(name);
    }

    int size = atoi(argv[1]);
    int nworkers = atoi(argv[2]);

    if (argc == 4)
    {
        std::string option(argv[3]);

        if (option == "-v")
        {
            verbose = true;
        }
        else
        {
            usage(name);
        }
    }

    std::cout << name << ": " << size << " " << nworkers << std::endl;

#ifdef _OPENMP
    omp_set_num_threads(nworkers);
#endif

    SquareMatrix target = generate(size);

    if (verbose)
    {
        std::cout << "Target:\n" << target.to_string() << std::endl;
    }

    auto begin = std::chrono::steady_clock::now();
    DecompResult result = decompose(target);
    auto end = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count();
    std::cout << "Elapsed time: " << elapsed << "ms" << std::endl;

    if (result.singular)
    {
        std::cout << "The generated matrix is singular." << std::endl;

        return 0;
    }
    else if (verbose)
    {
        std::cout << "Permutation:\n"
                  << result.permutation.to_string() << std::endl;
        std::cout << "Lower:\n" << result.lower.to_string() << std::endl;
        std::cout << "Upper:\n" << result.upper.to_string() << std::endl;
    }

    SquareMatrix residual =
        result.permutation * target - result.lower * result.upper;

    if (verbose)
    {
        std::cout << "Residual matrix:\n" << residual.to_string() << std::endl;
    }

    std::cout << "Norm of the residual matrix: " << residual.norm()
              << std::endl;

    return 0;
}
