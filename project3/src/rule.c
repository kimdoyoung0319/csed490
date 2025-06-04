#include "rule.h"
#include <stdio.h>
#include <string.h>

static int compute_living_neighbors(int *block, int size, int row, int col);
static void update_elem(int *elem, int num_living_neighbors);
static inline int *block_elem(int *block, int size, int row, int col);
static inline int block_value(int *block, int size, int row, int col);

/* Updates each element of the block according to the rule of Conway's game of life. */
void update_block(int *block, int size) {
    int prev[size * size];
    memcpy(prev, block, size * size * sizeof(int));

    int num_living_neighbors;

    for (int i = 1; i < size - 1; i++) {
        for (int j = 1; j < size - 1; j++) {
            num_living_neighbors = compute_living_neighbors(prev, size, i, j);
            update_elem(block_elem(block, size, i, j), num_living_neighbors);
        }
    }
}

/* Computes the number of living neighbors around the element (i, j). */
static int compute_living_neighbors(int *block, int size, int row, int col) {
    int count = 0;

    for (int i = row - 1; i <= row + 1; i++) {
        for (int j = col - 1; j <= col + 1; j++) {
            if (i == row && j == col)
                continue;

            if (block_value(block, size, i, j))
                count++;
        }
    }

    return count;
}

/* Updates the element pointed by the pointer elem, with respect to the number of living neighbors
 * around this element. */
static void update_elem(int *elem, int num_living_neighbors) {
    if (*elem && (num_living_neighbors == 2 || num_living_neighbors == 3))
        *elem = 1;
    else if (!*elem && num_living_neighbors == 3)
        *elem = 1;
    else
        *elem = 0;
}

/* Returns the pointer to the element (row, col) in the block. */
static inline int *block_elem(int *block, int size, int row, int col) {
    return block + row * size + col;
}

/* Returns the value of the element (row, col) in the block. */
static inline int block_value(int *block, int size, int row, int col) {
    return *block_elem(block, size, row, col);
}