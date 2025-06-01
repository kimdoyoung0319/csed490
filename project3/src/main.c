#include "board.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>

#define MASTER 0
#define MAX_NPROCS 100

int rank, size, grid, pad, nprocs, ngens, nghosts, coords[2], displs[MAX_NPROCS],
    counts[MAX_NPROCS];
struct board *board, *block;
MPI_Comm grid_comm;
MPI_Datatype block_t, row_t, col_t;

void initialize(int argc, char *argv[]);
void finalize();

void scatter_input();
void gather_output();

int living_neighbors(int y, int x);
void advance_gen();
void exchange_rows();

int main(int argc, char *argv[]) {
    initialize(argc, argv);
    scatter_input();
    print_subboard(board, size);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int gen = 0; gen < ngens; gen++) {
        printf("(%d, %d), generation #%d:\n", coords[0], coords[1], gen);
        MPI_Barrier(MPI_COMM_WORLD);
        print_board(block);
        printf("\n");
        exchange_rows();
        advance_gen();

        MPI_Barrier(MPI_COMM_WORLD);
    }

    gather_output();
    finalize();
}

void initialize(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0)
        scanf("%d\n%d\n%d\n", &size, &ngens, &nghosts);

    MPI_Bcast(&size, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&ngens, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&nghosts, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    grid = (int) sqrt((double) nprocs);

    MPI_Cart_create(MPI_COMM_WORLD, 2, (int[]) {grid, grid}, (int[]) {0, 0}, 0, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    int block_size = size / grid + 1;
    pad = block_size * grid - size;

    if (rank == 0)
        board = make_board(size + pad);
    block = make_board(block_size);

    MPI_Type_vector(get_size(block), get_size(block), size + pad, MPI_INT, &block_t);
    MPI_Type_create_resized(block_t, 0, sizeof(int), &block_t);
    MPI_Type_commit(&block_t);

    MPI_Type_vector(1, get_size(block), 0, MPI_INT, &row_t);
    MPI_Type_create_resized(row_t, 0, sizeof(int), &row_t);
    MPI_Type_commit(&row_t);

    MPI_Type_vector(get_size(block), 1, get_size(block), MPI_INT, &col_t);
    MPI_Type_create_resized(col_t, 0, sizeof(int), &col_t);
    MPI_Type_commit(&col_t);

    for (int i = 0; i < grid; i++) {
        for (int j = 0; j < grid; j++) {
            displs[i * grid + j] = (i * (size + pad) + j) * get_size(block);
            counts[i * grid + j] = 1;
        }
    }
}

void finalize() {
    destroy_board(board);
    destroy_board(block);

    MPI_Finalize();
}

void scatter_input() {
    if (rank == 0) {
        char ch;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                scanf("%c\n", &ch);
                set_elem(board, i, j, ch == '#');
            }
        }
    }

    MPI_Scatterv(get_arr(board), counts, displs, block_t, get_arr(block),
                 get_size(block) * get_size(block), MPI_INT, MASTER, MPI_COMM_WORLD);
}

void gather_output() {
    MPI_Gatherv(get_arr(block), get_size(block) * get_size(block), MPI_INT, get_arr(board), counts,
                displs, block_t, MASTER, MPI_COMM_WORLD);
}

int living_neighbors(int y, int x) {
    int count = 0;

    for (int i = y - 1; i <= y + 1; i++) {
        for (int j = x - 1; j <= x + 1; j++) {
            if (i == y && j == x)
                continue;

            if (get_elem(block, i, j))
                count++;
        }
    }

    return count++;
}

void advance_gen() {
    int there, here;
    struct board *next = make_board(get_size(block));

    for (int i = 0; i < get_size(block); i++) {
        for (int j = 0; j < get_size(block); j++) {
            here = get_elem(block, i, j);
            there = living_neighbors(i, j);

            if (here && (there == 2 || there == 3) || !here && there == 3)
                set_elem(next, i, j, 1);
        }
    }

    copy_board(block, next);
    destroy_board(next);
}

void exchange_rows() {
    int source, dest, block_size = get_size(block);
    int up[block_size], down[block_size], left[block_size], right[block_size];
    MPI_Status status;

    MPI_Cart_shift(grid_comm, 1, 1, &source, &dest);
    MPI_Send(get_row(block, get_size(block) - 1), 1, row_t, dest, 0, grid_comm);
    MPI_Recv(up, get_size(block), MPI_INT, source, MPI_ANY_TAG, grid_comm, &status);

    MPI_Cart_shift(grid_comm, 1, -1, &source, &dest);
    MPI_Send(get_row(block, get_size(block) - 1), 1, row_t, dest, 0, grid_comm);
    MPI_Recv(down, get_size(block), MPI_INT, source, MPI_ANY_TAG, grid_comm, &status);

    MPI_Cart_shift(grid_comm, 2, 1, &source, &dest);
    MPI_Send(get_row(block, get_size(block) - 1), 1, col_t, dest, 0, grid_comm);
    MPI_Recv(right, get_size(block), MPI_INT, source, MPI_ANY_TAG, grid_comm, &status);

    MPI_Cart_shift(grid_comm, 2, -1, &source, &dest);
    MPI_Send(get_row(block, get_size(block) - 1), 1, col_t, dest, 0, grid_comm);
    MPI_Recv(left, get_size(block), MPI_INT, source, MPI_ANY_TAG, grid_comm, &status);

    printf("(%d, %d), received from above:\n", coords[0], coords[1]);
    for (int i = 0; i < get_size(block); i++)
        printf("%s", up[i] ? "⬛" : "⬜");
    printf("\n\n");

    printf("(%d, %d), received from below:\n", coords[0], coords[1]);
    for (int i = 0; i < get_size(block); i++)
        printf("%s", down[i] ? "⬛" : "⬜");
    printf("\n\n");

    printf("(%d, %d), received from right:\n", coords[0], coords[1]);
    for (int i = 0; i < get_size(block); i++)
        printf("%s\n", right[i] ? "⬛" : "⬜");
    printf("\n\n");

    printf("(%d, %d), received from left:\n", coords[0], coords[1]);
    for (int i = 0; i < get_size(block); i++)
        printf("%s\n", left[i] ? "⬛" : "⬜");
    printf("\n\n");
}