#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

double *target, *lower, *upper;
size_t *permutation;
int nworkers;

void initialize(size_t size);
void generate(size_t size);
void run(size_t size);

void usage(const char *name) {
  std::cout << "usage: " << name << " matrix-size nworkers" << std::endl;
  exit(-1);
}

int main(int argc, char **argv) {
  const char *name = argv[0];

  if (argc < 3)
    usage(name);

  int size = atoi(argv[1]);

  nworkers = atoi(argv[2]);

  std::cout << name << ": " << size << " " << nworkers << std::endl;

  omp_set_num_threads(nworkers);

  generate(size);
  initialize(size);

  auto begin = std::chrono::steady_clock::now();
  run(size);
  auto end = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

  std::cout << "Elapsed time: " << elapsed.count() << "ms" << std::endl;

  return 0;
}

void initialize(size_t size) {
  lower = (double *)calloc(size * size, sizeof(double));
  upper = (double *)calloc(size * size, sizeof(double));
  permutation = (size_t *)calloc(size, sizeof(size_t));

  for (size_t i = 0; i < size; i++) {
    permutation[i] = i;
    lower[i * size + i] = 1.0l;
  }
}

void generate(size_t size) {
  std::random_device d;
  std::default_random_engine e(d());
  std::uniform_real_distribution dist(0.0l, 10.0l);

  target = (double *)malloc(sizeof(double) * size * size);

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      target[i * size + j] = dist(e);
    }
  }
};

void run(size_t size) {
#pragma omp parallel shared(target, lower, upper, permutation)
  for (size_t block = 0; block < size; block++) {
    size_t pivot = block;

    #pragma omp for
    for (size_t i = block; i < size; i++) {
      pivot =
          target[pivot * size + block] < target[i + size + block] ? i : pivot;
    }

    size_t temp = permutation[block];
    permutation[block] = permutation[pivot];
    permutation[block] = temp;

    #pragma omp for
    for (size_t i = 0; i < size; i++) {
      double temp = target[block * size + i];
      target[block * size + i] = target[pivot * size + i];
      target[pivot * size + i] = temp;
    }

    #pragma omp for
    for (size_t i = 0; i < block; i++) {
      double temp = lower[block * size + i];
      lower[block * size + i] = lower[pivot * size + i];
      lower[pivot * size + i] = temp;
    }

    upper[block * size + block] = target[block * size + block];

    #pragma omp for
    for (size_t i = block + 1; i < size; i++) {
      lower[i * size + block] =
          target[i * size + block] / upper[block * size + block];
      upper[block * size + i] = target[block * size + i];
    }

    #pragma omp for 
    for (size_t i = block + 1; i < size; i++) {
      for (size_t j = block + 1; j < size; j++) {
        target[i * size + j] -=
            lower[i * size + block] * upper[block * size + j];
      }
    }
  }
}