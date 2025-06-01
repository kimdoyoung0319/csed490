#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

MPI_Datatype block_t, col_t;
MPI_Comm grid_comm;

void read_input(int *buf, int width, int padded_width);
void print_board(int *board, int width);
void print_row(int *row, int length);
void advance_generation(int *board, int width);
int living_neighbors(int *board, int y, int x, int width);
void exchange_boundaries(int *board, int width);

int main(int argc, char *argv[]) {
    int rank, size, width, grid_width, final_gen, num_ghost;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    grid_width = (int) sqrt((double) size);

    if (rank == 0)
        scanf("%d\n%d\n%d\n", &width, &final_gen, &num_ghost);

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&final_gen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_ghost, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int block_width = width / grid_width + 1;
    int padded_width = grid_width * block_width;

    MPI_Type_vector(block_width, block_width, padded_width, MPI_INT, &block_t);
    MPI_Type_create_resized(block_t, 0, sizeof(int), &block_t);
    MPI_Type_commit(&block_t);

    MPI_Type_vector(block_width, 1, padded_width, MPI_INT, &col_t);
    MPI_Type_create_resized(col_t, 0, sizeof(int), &col_t);
    MPI_Type_commit(&col_t);

    int ndims[] = {grid_width, grid_width};
    int periods[] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, ndims, periods, 0, &grid_comm);
    printf("%d\n", rank);

    int *board;

    if (rank == 0) {
        board = (int *) calloc(padded_width * padded_width, sizeof(int));
        read_input(board, width, padded_width);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int *block = (int *) calloc(block_width * block_width, sizeof(int));
    int *displs = (int *) calloc(size, sizeof(int));
    int *counts = (int *) calloc(size, sizeof(int));

    for (int i = 0; i < grid_width; i++) {
        for (int j = 0; j < grid_width; j++) {
            displs[i * grid_width + j] = (i * padded_width + j) * block_width;
            counts[i * grid_width + j] = 1;
        }
    }

    MPI_Scatterv(board, counts, displs, block_t, block,
                 block_width * block_width, MPI_INT, 0, MPI_COMM_WORLD);

    for (int gen = 0; gen < final_gen; gen++) {
        advance_generation(block, block_width);
        exchange_boundaries(block, block_width);
        MPI_Barrier(MPI_COMM_WORLD);

        printf("Rank #%d:\n", rank);
        print_board(block, block_width);
        printf("\n");
    }

    MPI_Gatherv(block, block_width * block_width, MPI_INT, board, counts,
                displs, block_t, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(board);
    }

    free(block);
    free(displs);
    free(counts);

    MPI_Finalize();

    return 0;
}

void read_input(int *buf, int width, int padded_width) {
    char ch;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            scanf("%c\n", &ch);
            buf[i * padded_width + j] = (ch == '#');
        }
    }
}

void print_board(int *board, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++)
            printf("%s", board[i * width + j] ? "⬛" : "⬜");
        printf("\n");
    }
}

void print_row(int *row, int length) {
    for (int i = 0; i < length; i++)
        printf("%s", row[i] ? "⬛" : "⬜");
    printf("\n");
}

void advance_generation(int *board, int width) {
    int *next = (int *) calloc(width * width, sizeof(int));
    int up, down, left, right, neighbors;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            neighbors = living_neighbors(board, i, j, width);

            if (board[i * width + j] && (neighbors == 2 || neighbors == 3) ||
                !board[i * width + j] && neighbors == 3)
                next[i * width + j] = 1;
        }
    }

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++)
            board[i * width + j] = next[i * width + j];
    }

    free(next);
}

int living_neighbors(int *board, int y, int x, int width) {
    int count = 0;

    for (int i = y - 1; i <= y + 1; i++) {
        for (int j = x - 1; j <= x + 1; j++) {
            if (i == y && j == x)
                continue;

            if (i < 0 || i >= width || j < 0 || j >= width)
                continue;

            if (board[i * width + j])
                count++;
        }
    }

    return count++;
}

void exchange_boundaries(int *board, int width) {
    int source, dest, rank;
    int *our_row = board + width * (width - 1),
        *their_row = board + width * width;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Rank #%d, sending row:\n", rank);
    print_row(our_row, width);

    MPI_Cart_shift(grid_comm, 1, 1, &source, &dest);
    MPI_Sendrecv(our_row, width, MPI_INT, dest, 0, their_row, width, MPI_INT,
                 source, MPI_ANY_TAG, grid_comm, &status);
}