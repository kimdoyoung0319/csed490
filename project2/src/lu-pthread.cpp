#include "matrix.h"
#include "verify.cpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <vector>

void *find_pivot_row(void *args);
void *compute_elements(void *args);
void *update_target_matrix(void *args);
std::vector<size_t> lu_decomposition_parallel_improved(Matrix &A, Matrix &L,
                                                       Matrix &U, int n, int t);
void *worker_thread(void *args);

size_t num_threads, size, block;
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

/* LU decomposition algorithm implemented using pthreads. */
std::vector<size_t> lu_decomposition_parallel(Matrix &A, Matrix &L, Matrix &U,
                                              int n, int t)
{
    using std::swap;

    printf("Running Pthread version...\n");

    /* The permutation matrix. */
    std::vector<size_t> P(n);

    /* Thread handles. */
    pthread_t threads[t];

    /* Makes a copy of the target matrix A. */
    Matrix B(A);

    /* Initializes global variables that will be used across threads. */
    target = &B;
    lower = &L;
    upper = &U;
    num_threads = t;
    size = n;

    for (size_t i = 0; i < n; i++)
    {
        P[i] = i;
        L[i][i] = 1.0;
    }

    for (size_t k = 0; k < n; k++)
    {
        block = k;
        size_t pivot = k;

        /* Finds pivot row. */
        for (size_t i = k; i < n; i++)
        {
            if (std::abs(B[pivot][k]) < std::abs(B[i][k]))
                pivot = i;
        }

        if (k != pivot)
        {
            swap(P[k], P[pivot]);
            swap(B[k], B[pivot]);
            swap_ranges(L[k].begin(), L[k].begin() + k, L[pivot].begin());
        }

        U[k][k] = B[k][k];

        for (size_t id = 0; id < t; id++)
            pthread_create(&threads[id], nullptr, compute_elements, (void *)id);

        size_t start = (n - k - 1) / t * t + k + 1;

        for (size_t i = start; i < n; i++)
        {
            L[i][k] = B[i][k] / U[k][k];
            U[k][i] = B[k][i];
        }

        for (size_t id = 0; id < t; id++)
            pthread_join(threads[id], NULL);

        for (size_t id = 0; id < t; id++)
            pthread_create(&threads[id], nullptr, update_target_matrix,
                           (void *)id);

        for (size_t i = start; i < n; i++)
        {
            for (size_t j = k + 1; j < n; j++)
                B[i][j] -= L[i][k] * U[k][j];
        }

        for (size_t id = 0; id < t; id++)
            pthread_join(threads[id], NULL);
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
#ifdef PARALLEL
    std::vector<size_t> P = lu_decomposition_parallel_improved(A, L, U, n, t);
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

    std::cout << "Elapsed time: " << elapsed.count() << std::endl;
    std::cout << "L2,1 norm: " << verify(P, A, L, U, n) << std::endl;

    return 0;
}

/* Finds the maximum row (i.e. the row with maximum first element.) within the
   range that is assigned to this thread. */
void *find_pivot_row(void *arg)
{
    size_t id = (size_t)arg;
    size_t loop_per_thread = (size - block) / num_threads;

    size_t start = loop_per_thread * id + block;
    size_t end = loop_per_thread * (id + 1) + block;
    size_t max_index = start;

    Matrix &A = *target;

    for (size_t i = start; i < end; i++)
    {
        if (std::abs(A[max_index][block]) < std::abs(A[i][block]))
            max_index = i;
    }

    pthread_exit((void *)max_index);
}

/* Computes elements within the thread-specific range. */
void *compute_elements(void *arg)
{
    size_t id = (size_t)arg;
    size_t loop_per_thread = (size - block - 1) / num_threads;

    size_t start = loop_per_thread * id + (block + 1);
    size_t end = loop_per_thread * (id + 1) + (block + 1);

    Matrix &A = *target, &L = *lower, &U = *upper;

    for (size_t i = start; i < end; i++)
    {
        L[i][block] = A[i][block] / U[block][block];
        U[block][i] = A[block][i];
    }

    pthread_exit(nullptr);
}

/* Updates rows of the target matrix A within the thread-specific range. */
void *update_target_matrix(void *arg)
{
    size_t id = (size_t)arg;
    size_t loop_per_thread = (size - block - 1) / num_threads;

    size_t start = loop_per_thread * id + (block + 1);
    size_t end = loop_per_thread * (id + 1) + (block + 1);

    Matrix &A = *target, &L = *lower, &U = *upper;

    for (size_t i = start; i < end; i++)
    {
        for (size_t j = block + 1; j < size; j++)
            A[i][j] -= L[i][block] * U[block][j];
    }

    pthread_exit(nullptr);
}

std::vector<size_t> lu_decomposition_parallel_improved(Matrix &A, Matrix &L,
                                                       Matrix &U, int n, int t)
{
    using std::swap;
    using std::swap_ranges;

    printf("Running pthread version...\n");

    std::vector<size_t> P(n);

    pthread_t threads[t];

    Matrix B(A);

    target = &B;
    lower = &L;
    upper = &U;
    num_threads = t;
    size = n;

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

    for (size_t k = 0; k < n; k++)
    {
        size_t start = (n - k - 1) / t * t + k + 1, pivot = k;

        for (size_t i = k; i < n; i++)
        {
            if (std::abs(B[pivot][k]) < std::abs(B[i][k]))
                pivot = i;
        }

        if (k != pivot)
        {
            swap(P[k], P[pivot]);
            swap(B[k], B[pivot]);
            swap_ranges(L[k].begin(), L[k].begin() + k, L[pivot].begin());
        }

        U[k][k] = B[k][k];

        pthread_barrier_wait(&barrier1);

        for (size_t i = start; i < n; i++)
        {
            L[i][k] = B[i][k] / U[k][k];
            U[k][i] = B[k][i];
        }

        pthread_barrier_wait(&barrier2);

        for (size_t i = start; i < n; i++)
        {
            for (size_t j = k + 1; j < n; j++)
                B[i][j] -= L[i][k] * U[k][j];
        }

        pthread_barrier_wait(&barrier3);
    }

    pthread_barrier_destroy(&barrier1);
    pthread_barrier_destroy(&barrier2);
    pthread_barrier_destroy(&barrier3);

    return P;
}

void *worker_thread(void *arg)
{
    size_t id = (size_t)arg;

    Matrix &B = *target;
    Matrix &L = *lower;
    Matrix &U = *upper;

    for (size_t k = 0; k < size; k++)
    {
        size_t loop_per_thread = (size - k - 1) / num_threads;
        size_t start = loop_per_thread * id + (k + 1);
        size_t end = loop_per_thread * (id + 1) + (k + 1);

        pthread_barrier_wait(&barrier1);

        for (size_t i = start; i < end; i++)
        {
            L[i][k] = B[i][k] / U[k][k];
            U[k][i] = B[k][i];
        }

        pthread_barrier_wait(&barrier2);

        for (size_t i = start; i < end; i++)
        {
            for (size_t j = k + 1; j < size; j++)
                B[i][j] -= L[i][k] * U[k][j];
        }

        pthread_barrier_wait(&barrier3);
    }

    pthread_exit(NULL);
}