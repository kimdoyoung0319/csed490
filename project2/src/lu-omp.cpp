#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <omp.h>

void print_matrix(const std::vector<std::vector<double>>& mat, const char* name) {
    printf("%s:\n", name);
    for (const auto& row : mat) {
        for (double val : row)
            printf("%.4e ", val);
        printf("\n");
    }
}

void lu_decomposition_omp(std::vector<std::vector<double>>& A,
                          std::vector<std::vector<double>>& L,
                          std::vector<std::vector<double>>& U,
                          int n, int t) {
    // TODO: implement LU decomposition using OpenMP
    printf("Running OpenMP version...\n");
    #pragma omp parallel shared(a, l, u, p, size)
    for (size_t k = 0; k < size; k++) {
        size_t pivot = k;

#pragma omp for
        for (size_t i = k; i < size; i++) {
            if (abs(a[pivot * size + k]) < abs(a[i * size + k]))
                pivot = i;
        }

        if (k != pivot) {
            std::swap(p[k], p[pivot]);
            std::swap_ranges(a + k * size, a + k * size + size,
                             a + pivot * size);
            std::swap_ranges(pa + k * size, pa + k * size + size,
                             pa + pivot * size);
            std::swap_ranges(l + k * size, l + k * size + k, l + pivot * size);
        }

        u[k * size + k] = a[k * size + k];

#pragma omp for
        for (size_t i = k + 1; i < size; i++) {
            l[i * size + k] = a[i * size + k] / u[k * size + k];
            u[k * size + i] = a[k * size + i];
        }

#pragma omp for
        for (size_t i = k + 1; i < size; i++) {
            for (size_t j = k + 1; j < size; j++)
                a[i * size + j] -= l[i * size + k] * u[k * size + j];
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <n> <r> <t> <p>\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int r = atoi(argv[2]);
    int t = atoi(argv[3]);
    int p = atoi(argv[4]);

    srand(r);
    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = rand() / (double)RAND_MAX;

    lu_decomposition_omp(A, L, U, n, t);

    if (p == 1) {
        print_matrix(L, "L");
        print_matrix(U, "U");
        print_matrix(A, "A");
    }

    return 0;
}
