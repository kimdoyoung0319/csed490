#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

double *a, *l, *u;
size_t *p;
int nworkers;

void initialize(size_t size);
void generate(size_t size);
void run(size_t size);
void verify(size_t size);
void clean();
void print(double *a, size_t size);

void usage(const char *name) {
    std::cout << "usage: " << name << " matrix-size nworkers" << std::endl;
    exit(-1);
}

int main(int argc, char **argv) {
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;
    using std::chrono::steady_clock;

    const char *name = argv[0];

    if (argc < 3)
        usage(name);

    int size = atoi(argv[1]);

    nworkers = atoi(argv[2]);

    std::cout << name << ": " << size << " " << nworkers << std::endl;

    omp_set_num_threads(nworkers);

    generate(size);
    initialize(size);

    auto begin = steady_clock::now();
    run(size);
    auto end = steady_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - begin);

    std::cout << "A: " << std::endl;
    print(a, size);
    std::cout << "L: " << std::endl;
    print(l, size);
    std::cout << "U: " << std::endl;
    print(u, size);

    std::cout << "Elapsed time: " << elapsed.count() << "ms" << std::endl;
    verify(size);

    clean();

    return 0;
}

void initialize(size_t size) {
    l = new double[size * size];
    u = new double[size * size];
    p = new size_t[size];

    for (size_t i = 0; i < size; i++) {
        p[i] = i;
        l[i * size + i] = 1.0l;
    }
}

void generate(size_t size) {
    std::random_device d;
    std::default_random_engine e(d());
    std::uniform_real_distribution dist(0.0l, 10.0l);

    a = new double[size * size];

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            a[i * size + j] = dist(e);
        }
    }
};

void run(size_t size) {
#pragma omp parallel shared(a, l, u, p)
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

void verify(size_t size) {
    double norm = 0.0;
    double *r = new double[size * size];
    double *lu = new double[size * size];

#pragma omp parallel shared(p, a, l, u, r, norm)
    {
#pragma omp for
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                for (size_t k = 0; k < size; k++) {
                    lu[i * size + j] += l[i * size + k] * u[k * size + j];
                }
            }
        }

        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                r[i * size + j] = a[i * size + j] - lu[i * size + j];
            }
        }

#pragma omp for
        for (size_t j = 0; j < size; j++) {
            double sum = 0.0;

            for (size_t i = 0; i < size; i++) {
                sum += r[i * size + j] * r[i * size + j];
            }

#pragma omp critical
            {
                norm += sqrt(sum);
            }
        }
    }

#pragma omp single
    {
        std::cout << "Residual: " << std::endl;
        print(r, size);
        std::cout << "PA: " << std::endl;
        print(a, size);
        std::cout << "LU: " << std::endl;
        print(lu, size);
    }

    std::cout << "L2,1 norm of the residual matrix: " << norm << std::endl;
    delete[] r;
    delete[] lu;
}

void clean() {
    delete[] a;
    delete[] l;
    delete[] u;
    delete[] p;
}

void print(double *a, size_t size) {
    std::cout << std::fixed << std::setprecision(2);

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++)
            std::cout << std::setw(5) << a[i * size + j] << ' ';

        std::cout << '\n';
    }

    std::cout << '\n';
}