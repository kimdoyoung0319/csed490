#include "matrix.h"
#include "verify.cpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <vector>

void *worker_thread(void *args);

size_t num_threads, size;
Matrix *target, *lower, *upper;
pthread_barrier_t barrier1, barrier2, barrier3;

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

/* Sequential version of LU decomposition algorithm. */
std::vector<size_t> lu_decomposition_seq(Matrix &A, Matrix &L, Matrix &U, int n)
{
    using std::swap;
    using std::swap_ranges;

    printf("Running sequential version...\n");

    /* Permutation matrix, which is actually represented as a vector<int>. */
    std::vector<size_t> P(n);

    /* The copy of matrix A. */
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

        for (size_t i = k + 1; i < n; i++)
        {
            L[i][k] = B[i][k] / U[k][k];
            U[k][i] = B[k][i];
        }

        for (size_t i = k + 1; i < n; i++)
        {
            for (size_t j = k + 1; j < n; j++)
                B[i][j] -= L[i][k] * U[k][j];
        }
    }

    return P;
}

/* Parallel version of LU decomposition algorithm implemented using pthreads. */
std::vector<size_t> lu_decomposition_parallel(Matrix &A, Matrix &L, Matrix &U,
                                              int n, int t)
{
    using std::swap;
    using std::swap_ranges;

    printf("Running pthread version...\n");

    /* Permutation vector. */
    std::vector<size_t> P(n);

    /* Thread handles. */
    pthread_t threads[t];

    /* The copy of the original target matrix A. */
    Matrix B(A);

    /* Initializes global variables that will be shared throughout the worker
       threads. */
    target = &B;
    lower = &L;
    upper = &U;
    num_threads = t;
    size = n;

    /* Initializes barriers, permutation vector, lower matrix, and creates
       worker threads.*/
    pthread_barrier_init(&barrier1, NULL, t + 1);
    pthread_barrier_init(&barrier2, NULL, t + 1);
    pthread_barrier_init(&barrier3, NULL, t + 1);

    for (size_t i = 0; i < n; i++)
    {
        P[i] = i;
        L[i][i] = 1.0;
    }

    for (size_t id = 0; id < t; id++)
        pthread_create(&threads[id], NULL, worker_thread, (void *)id);

    /* Main loop over blocks. */
    for (size_t k = 0; k < n; k++)
    {
        /* The main thread should compute residual loops. After distributing
           ((n - k - 1) / t) loops per thread, there are some loops that are not
           performed by worker threads. `start` is the starting index of such
           loops. */
        size_t start = (n - k - 1) / t * t + k + 1, pivot = k;

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

        /* Alerts other threads the main thread has done its job. */
        pthread_barrier_wait(&barrier1);

        /* Performs the residual loops for lower and upper matrices. */
        for (size_t i = start; i < n; i++)
        {
            L[i][k] = B[i][k] / U[k][k];
            U[k][i] = B[k][i];
        }

        /* Waits for the other threads to complete their work. */
        pthread_barrier_wait(&barrier2);

        /* Performs the residual loops for the target matrix. */
        for (size_t i = start; i < n; i++)
        {
            for (size_t j = k + 1; j < n; j++)
                B[i][j] -= L[i][k] * U[k][j];
        }

        /* Waits for the other threads to complete their work. */
        pthread_barrier_wait(&barrier3);
    }

    pthread_barrier_destroy(&barrier1);
    pthread_barrier_destroy(&barrier2);
    pthread_barrier_destroy(&barrier3);

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
#ifdef PARALLEL
    std::vector<size_t> P = lu_decomposition_parallel(A, L, U, n, t);
#else
    std::vector<size_t> P = lu_decomposition_seq(A, L, U, n);
#endif
    auto end = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    if (p)
    {
        print_matrix(L, "L");
        print_matrix(U, "U");
        print_matrix(A, "A");
    }

    std::cout << elapsed.count() << std::endl;
    std::cout << verify(P, A, L, U, n) << std::endl;

    return 0;
}

/* The function that will be executed by each worker thread. */
void *worker_thread(void *arg)
{
    /* Thread id is given by its argument. */
    size_t id = (size_t)arg;

    Matrix &B = *target;
    Matrix &L = *lower;
    Matrix &U = *upper;

    /* Main thread over the blocks. */
    for (size_t k = 0; k < size; k++)
    {
        /* Computes its portion of loop with respects to its id. */
        size_t loop_per_thread = (size - k - 1) / num_threads;
        size_t start = loop_per_thread * id + (k + 1);
        size_t end = loop_per_thread * (id + 1) + (k + 1);

        /* Waits for the main thread to calculate pivot and swap rows. */
        pthread_barrier_wait(&barrier1);

        /* Computes its portion of matrix elements. */
        for (size_t i = start; i < end; i++)
        {
            L[i][k] = B[i][k] / U[k][k];
            U[k][i] = B[k][i];
        }

        /* Waits the other threads to complete their work. */
        pthread_barrier_wait(&barrier2);

        /* Updates its portion of target matrix. */
        for (size_t i = start; i < end; i++)
        {
            for (size_t j = k + 1; j < size; j++)
                B[i][j] -= L[i][k] * U[k][j];
        }

        /* Waits the other threads to complete their work. */
        pthread_barrier_wait(&barrier3);
    }

    pthread_exit(NULL);
}