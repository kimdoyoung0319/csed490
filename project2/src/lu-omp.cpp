#include "matrix.h"
#include "verify.cpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

void print_matrix(const Matrix &mat, const char *name)
{
    printf("%s:\n", name);
    for (const auto &row : mat)
    {
        for (double val : row)
            printf("%.4e ", val);
        printf("\n");
    }
}

/* LU decomposition algorithm implemented using OpenMP. */
std::vector<size_t> lu_decomposition_omp(Matrix &A, Matrix &L, Matrix &U, int n,
                                         int t)
{
    using std::swap;
    using std::swap_ranges;

    printf("Running OpenMP version...\n");

    /* Permutation matrix, which is actually represented as a vector<int>. */
    std::vector<size_t> P(n);

    /* The copy of the original target matrix A. */
    Matrix B(A);

    /* Initializes the permutation matrix and the lower matrix. */
    for (size_t i = 0; i < n; i++)
    {
        P[i] = i;
        L[i][i] = 1.0;
    }

    /* Main loop over blocks. */
    for (size_t k = 0; k < n; k++)
    {
        size_t pivot = k;

        /* Finds pivot row. */
        for (size_t i = k; i < n; i++)
        {
            if (std::abs(B[pivot][k]) < std::abs(B[i][k]))
                pivot = i;
        }

        /* Peforms swap for each matrices. */
        if (k != pivot)
        {
            swap(P[k], P[pivot]);
            swap(B[k], B[pivot]);
            swap_ranges(L[k].begin(), L[k].begin() + k, L[pivot].begin());
        }

        /* Computes the lower and upper matrices with respect to the original
           matrix. */
        U[k][k] = B[k][k];

#pragma omp parallel for shared(L, U, B, k) num_threads(t)
        for (size_t i = k + 1; i < n; i++)
        {
            L[i][k] = B[i][k] / U[k][k];
            U[k][i] = B[k][i];
        }

#pragma omp parallel for shared(L, U, B, k) num_threads(t)
        for (size_t i = k + 1; i < n; i++)
        {
            for (size_t j = k + 1; j < n; j++)
                B[i][j] -= L[i][k] * U[k][j];
        }
    }

    return P;
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <n> <r> <t> <p>\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int r = atoi(argv[2]);
    int t = atoi(argv[3]);
    int p = atoi(argv[4]);

    srand(r);
    Matrix A(n, Row(n));
    Matrix L(n, Row(n, 0.0));
    Matrix U(n, Row(n, 0.0));

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
            A[i][j] = rand() / (double)RAND_MAX;
    }

    auto begin = std::chrono::steady_clock::now();
    std::vector<size_t> P = lu_decomposition_omp(A, L, U, n, t);
    auto end = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    if (p)
    {
        print_matrix(L, "L");
        print_matrix(U, "U");
        print_matrix(A, "A");
    }

    std::cout << "Elapsed time: " << elapsed.count() << std::endl;
    std::cout << "L2,1 norm: " << verify(P, A, L, U, n) << std::endl;

    return 0;
}
