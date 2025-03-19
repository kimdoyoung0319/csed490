#include <iostream>
#include <omp.h>

#include "decomp.h"
#include "gen.h"

void usage(const char *name)
{
    std::cout << "Usage: " << name << " matrix-size nworkers [options]\n"
              << "Options:\n"
              << "-v  Prints generated matrix and decomposed matrices."
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

    DecompResult result = decompose(target);

    if (result.singular && verbose)
    {
        std::cout << "The generated matrix is sigular." << std::endl;
    }
    else if (verbose)
    {
        std::cout << "Lower:\n" << result.lower.to_string() << std::endl;
        std::cout << "Upper:\n" << result.upper.to_string() << std::endl;
    }

    return 0;
}
