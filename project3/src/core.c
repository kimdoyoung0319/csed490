/* TODO: Add ghosts. */
#include "core.h"
#include "rule.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int rank;       // The rank of this process.
int num_procs;  // The number of processes in the global communicator.
int num_ghosts; // The number of ghost row/columns.
int num_gens;   // The number of generations to be executed.
int board_size; // The width of the square input board.
int block_size; // The width of a square block.
int grid_size;  // The number of blocks in each column or row.
int pad_size;   // The number of padded elements of the board.

/* Notice that the board_size and block_size may be inconsistent with the actual memory layout due
 * to the padding around the board or a block. For the entire board, we need to consider padding
 * with the board_size. For a block, we need to consider buffer portion for receiving data from
 * adjacent processes. */

int *board; // The entire board, which is undivided. Only used in the master process.
int *block; // A portion of board, for which this process is responsible.

MPI_Datatype board_t; // Datatype for scattering and gathering the entire board.
MPI_Datatype block_t; // Datatype for sending, or receiving a distributed block.
MPI_Datatype row_t;   // Datatype for a row of a block.
MPI_Datatype col_t;   // Datatype for a column of a block.

MPI_Comm cart; // Communicator for Cartesian topology.

static void read_basic();
static void read_board();
static void print_board();
static void print_block();
static void exchange();
static int *block_elem(int i, int j);
static void cart_shift_diagonal(int direction, int disp, int *source, int *dest);

/* Initializes the MPI environment, and sets the global variables for this module.*/
void initialize(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // grid_size is set to the square root of num_procs since if there are n^2 processes, then the
    // grid would be in size of n * n.
    grid_size = (int) sqrt((double) num_procs);

    MPI_Cart_create(MPI_COMM_WORLD, 2, (int[]) {grid_size, grid_size}, (int[]) {0, 0}, 0, &cart);

    read_basic();

    // Since the board size might not be divisible by the grid size, it adds padding at the end of
    // each row and column. Calculate the size of the padding.
    block_size = board_size / grid_size + 1;
    pad_size   = block_size * grid_size - board_size;

    if (rank == 0) {
        board = (int *) calloc((board_size + pad_size) * (board_size + pad_size), sizeof(int));
    }

    // Here, additional two elements are added to the block since we need the buffer to receive data
    // into.
    block = (int *) calloc((block_size + 2) * (block_size + 2), sizeof(int));

    MPI_Type_vector(block_size, block_size, board_size + pad_size, MPI_INT, &board_t);
    MPI_Type_create_resized(board_t, 0, sizeof(int), &board_t);
    MPI_Type_commit(&board_t);

    MPI_Type_vector(block_size, block_size, block_size + 2, MPI_INT, &block_t);
    MPI_Type_create_resized(block_t, 0, sizeof(int), &block_t);
    MPI_Type_commit(&block_t);

    MPI_Type_vector(1, block_size, 0, MPI_INT, &row_t);
    MPI_Type_create_resized(row_t, 0, sizeof(int), &row_t);
    MPI_Type_commit(&row_t);

    MPI_Type_vector(block_size, 1, block_size + 2, MPI_INT, &col_t);
    MPI_Type_create_resized(col_t, 0, sizeof(int), &col_t);
    MPI_Type_commit(&col_t);

    // We are ready to read board data from stdin, so we do so.
    read_board();
}

/* Cleans up the used resources, finalizes the MPI environment. */
void finalize() {
    if (board != NULL) {
        free(board);
    }

    if (block != NULL) {
        free(block);
    }

    MPI_Type_free(&board_t);
    MPI_Type_free(&row_t);
    MPI_Type_free(&col_t);

    MPI_Comm_free(&cart);

    MPI_Finalize();
}

/* Actually executes the game according to the rule. */
void execute() {
    int coords[2];

    for (int gen = 0; gen < num_gens; gen++) {
        if (rank == 1) {
            printf("[#%d]\n", gen);
            MPI_Cart_coords(cart, rank, 2, coords);
            printf("(%d, %d)\n", coords[0], coords[1]);
            print_block();
        }

        exchange();
        update_block(block, block_size);
    }

    int displs[num_procs];
    int counts[num_procs];

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            displs[i * grid_size + j] = (i * (board_size + pad_size) + j) * block_size;
            counts[i * grid_size + j] = 1;
        }
    }

    MPI_Gatherv(block_elem(1, 1), 1, block_t, board, counts, displs, board_t, 0, MPI_COMM_WORLD);
    print_board();
}

/* Actually execute the game according to the rule, waiting 1 second for each generation. */
void execute_wait() {
    int displs[num_procs];
    int counts[num_procs];

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            displs[i * grid_size + j] = (i * (board_size + pad_size) + j) * block_size;
            counts[i * grid_size + j] = 1;
        }
    }

    for (int gen = 0; gen < num_gens; gen++) {
        MPI_Gatherv(block_elem(1, 1), 1, block_t, board, counts, displs, board_t, 0,
                    MPI_COMM_WORLD);

        if (rank == 0) {
            printf("\e[1;1H\e[2J");
            printf("[%d]\n", gen);
        }

        print_board();
        exchange();
        update_block(block, block_size);

        float start = MPI_Wtime();
        while (MPI_Wtime() - start < 1.0)
            ;
    }

    MPI_Gatherv(block_elem(1, 1), 1, block_t, board, counts, displs, board_t, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\e[1;1H\e[2J");
        printf("[%d]\n", num_gens);
    }

    print_board();
}

/* Reads the basic information from the user, initializes global varibales for the board size,
 * the number of generations and ghosts. */
static void read_basic() {
    if (rank == 0) {
        scanf("%d\n%d\n%d\n", &board_size, &num_gens, &num_ghosts);
    }

    MPI_Bcast(&board_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_gens, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_ghosts, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

/* Reads the whole board from the stdin, distributes them to the processes in the world
 * communicator.*/
static void read_board() {
    if (rank == 0) {
        char ch;

        // Read each character from the stdin, store the result to the board.
        for (int i = 0; i < board_size; i++) {
            for (int j = 0; j < board_size; j++) {
                scanf("%c\n", &ch);
                board[i * (board_size + pad_size) + j] = (ch == '#');
            }
        }
    }

    int displs[num_procs];
    int counts[num_procs];

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            displs[i * grid_size + j] = (i * (board_size + pad_size) + j) * block_size;
            counts[i * grid_size + j] = 1;
        }
    }

    MPI_Scatterv(board, counts, displs, board_t, block_elem(1, 1), 1, block_t, 0, MPI_COMM_WORLD);
}

/* Prints the whole board into the stdin. Only the master process with rank 0 prints. Others will
 * ignore this function call since they do not have a board to print. */
static void print_board() {
    if (rank != 0) {
        return;
    }

    for (int i = 0; i < board_size; i++) {
        for (int j = 0; j < board_size; j++) {
            printf("%s", board[i * (board_size + pad_size) + j] ? "⬛" : "⬜");
        }
        printf("\n");
    }

    printf("\n");
}

/* Prints the distributed block for debugging purposes. */
static void print_block() {
    for (int i = 0; i < block_size + 2; i++) {
        for (int j = 0; j < block_size + 2; j++) {
            printf("%s", block[i * (block_size + 2) + j] ? "⬛" : "⬜");
        }

        printf("\n");
    }

    printf("\n");
}

/* Exchanges uppermost and lowermost rows, leftmost and rightmost columns, and elements in the
 * corners with adjacent processes in the Cartesian topology. */
static void exchange() {
    static const int horizontal = 0;
    static const int vertical   = 1;
    static const int downright  = 1;
    static const int downleft   = -1;
    static const int forward    = 1;
    static const int backward   = -1;

    int source, dest;

    // I hate my life.
    MPI_Cart_shift(cart, horizontal, forward, &source, &dest);
    MPI_Sendrecv(block_elem(block_size, 1), 1, row_t, dest, 0, block_elem(0, 1), 1, row_t, source,
                 0, cart, MPI_STATUS_IGNORE);

    MPI_Cart_shift(cart, horizontal, backward, &source, &dest);
    MPI_Sendrecv(block_elem(1, 1), 1, row_t, dest, 0, block_elem(block_size + 1, 1), 1, row_t,
                 source, 0, cart, MPI_STATUS_IGNORE);

    MPI_Cart_shift(cart, vertical, forward, &source, &dest);
    MPI_Sendrecv(block_elem(1, block_size), 1, col_t, dest, 0, block_elem(1, 0), 1, col_t, source,
                 0, cart, MPI_STATUS_IGNORE);

    MPI_Cart_shift(cart, vertical, backward, &source, &dest);
    MPI_Sendrecv(block_elem(1, 1), 1, col_t, dest, 0, block_elem(1, block_size + 1), 1, col_t,
                 source, 0, cart, MPI_STATUS_IGNORE);

    cart_shift_diagonal(downright, forward, &source, &dest);
    MPI_Sendrecv(block_elem(block_size, block_size), 1, MPI_INT, dest, 0, block_elem(0, 0), 1,
                 MPI_INT, source, 0, cart, MPI_STATUS_IGNORE);

    cart_shift_diagonal(downright, backward, &source, &dest);
    MPI_Sendrecv(block_elem(1, 1), 1, MPI_INT, dest, 0, block_elem(block_size + 1, block_size + 1),
                 1, MPI_INT, source, 0, cart, MPI_STATUS_IGNORE);

    cart_shift_diagonal(downleft, forward, &source, &dest);
    MPI_Sendrecv(block_elem(block_size, 1), 1, MPI_INT, dest, 0, block_elem(0, block_size + 1), 1,
                 MPI_INT, source, 0, cart, MPI_STATUS_IGNORE);

    cart_shift_diagonal(downleft, backward, &source, &dest);
    MPI_Sendrecv(block_elem(1, block_size), 1, MPI_INT, dest, 0, block_elem(block_size + 1, 0), 1,
                 MPI_INT, source, 0, cart, MPI_STATUS_IGNORE);
}

/* Returns the address element (row, col) of the block of this process. The row and column indexes
 * are relative to the actual padded block, not the original one. */
static int *block_elem(int row, int col) { return block + row * (block_size + 2) + col; }

/* Returns the shifted source and destination rank for diagonal shift within a Cartesian topology.
 * For direction > 0, it performs shift toward the downright direction. For direction < 0, it
 * performs shift toward the downleft direction. If disp < 0, the result direction is reversed. */
static void cart_shift_diagonal(int direction, int disp, int *source, int *dest) {
    int coords[2], source_coords[2], dest_coords[2];

    MPI_Cart_coords(cart, rank, 2, coords);

    if (direction > 0) {
        source_coords[0] = coords[0] - disp;
        source_coords[1] = coords[1] - disp;

        dest_coords[0] = coords[0] + disp;
        dest_coords[1] = coords[1] + disp;
    } else {
        source_coords[0] = coords[0] - disp;
        source_coords[1] = coords[1] + disp;

        dest_coords[0] = coords[0] + disp;
        dest_coords[1] = coords[1] - disp;
    }

    int source_out_of_bound = source_coords[0] < 0 || source_coords[0] >= grid_size ||
                              source_coords[1] < 0 || source_coords[1] >= grid_size;

    int dest_out_of_bound = dest_coords[0] < 0 || dest_coords[0] >= grid_size ||
                            dest_coords[1] < 0 || dest_coords[1] >= grid_size;

    if (source_out_of_bound)
        *source = MPI_PROC_NULL;
    else
        MPI_Cart_rank(cart, source_coords, source);

    if (dest_out_of_bound)
        *dest = MPI_PROC_NULL;
    else
        MPI_Cart_rank(cart, dest_coords, dest);
}