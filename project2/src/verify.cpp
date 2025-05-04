#include "matrix.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

/* Calculates L2,1 norm of PA - LU. */
double verify(std::vector<size_t> P, Matrix &A, Matrix &L, Matrix &U, size_t n)
{
    /* The residual matrix R = PA - LU. */
    Matrix R(n, Row(n));
    Matrix LU(n, Row(n));
    Matrix PA(n, Row(n));

    /* Computes PA. */
    for (size_t i = 0; i < n; i++)
        PA[i] = A[P[i]];

    /* Computes LU. */
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            for (size_t k = 0; k < n; k++)
                LU[i][j] += L[i][k] * U[k][j];
        }
    }

    /* Computes R. */
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
            R[i][j] = PA[i][j] - LU[i][j];
    }

    double norm = 0.0;

    /* Computes the sum of Euclidean norms of jth column, which is L2,1 norm. */
    for (size_t j = 0; j < n; j++)
    {
        double sum = 0.0;

        for (size_t i = 0; i < n; i++)
            sum += R[i][j] * R[i][j];

        norm += sqrt(sum);
    }

    return norm;
}