#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int a = 1;
    int b = 2;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Sendrecv(&b, 1, MPI_INT, 1, 0, &a, 1, MPI_INT, MPI_PROC_NULL, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    printf("[%d] a = %d\n", rank, a);

    MPI_Finalize();

    return 0;
}