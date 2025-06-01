#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

double *a, *pa, *l, *u;
size_t size;
size_t *p;
int nworkers;

void initialize();
void generate();
void run();
void verify();
void clean();
void print(double *a);

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

    size = atoi(argv[1]);

    nworkers = atoi(argv[2]);

    std::cout << name << ": " << size << " " << nworkers << std::endl;

    omp_set_num_threads(nworkers);

    generate();
    initialize();

    auto begin = steady_clock::now();
    run();
    auto end = steady_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - begin);

    std::cout << elapsed.count() << std::endl;
    // verify(size);

    clean();

    return 0;
}

void initialize() {
    l = new double[size * size];
    u = new double[size * size];
    p = new size_t[size];

    for (size_t i = 0; i < size; i++) {
        p[i] = i;
        l[i * size + i] = 1.0l;
    }
}

void generate() {
    std::random_device d;
    std::default_random_engine e(d());
    std::uniform_real_distribution dist(0.0l, 10.0l);

    a = new double[size * size];
    pa = new double[size * size];

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            a[i * size + j] = dist(e);
            pa[i * size + j] = a[i * size + j];
        }
    }
};

void run() {
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

void verify() {
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
                r[i * size + j] = pa[i * size + j] - lu[i * size + j];
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

    std::cout << "L2,1 norm of the residual matrix: " << norm << std::endl;
    delete[] r, lu;
}

void clean() { delete[] a, pa, l, u, p; }

void print(double *a) {
    std::cout << std::fixed << std::setprecision(2);

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++)
            std::cout << std::setw(5) << a[i * size + j] << ' ';

        std::cout << '\n';
    }

    std::cout << '\n';
}