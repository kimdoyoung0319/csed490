#include "board.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

inline static int is_board_valid(struct board *board);
inline static int is_index_valid(struct board *board, int row, int col);

struct board *make_board(int size) {
    struct board *new = (struct board *) malloc(sizeof(struct board));

    if (new == NULL)
        return NULL;

    new->size = size;
    new->arr = (int *) calloc(size * size, sizeof(int));

    if (new->arr == NULL) {
        free(new);
        return NULL;
    }

    return new;
}

void destroy_board(struct board *board) {
    if (board == NULL)
        return;

    if (board->arr != NULL)
        free(board->arr);

    free(board);
}

inline int get_size(struct board *board) {
    if (board == NULL)
        return -1;

    return board->size;
}

inline int *get_arr(struct board *board) {
    if (!is_board_valid(board))
        return NULL;

    return board->arr;
}

inline int *get_row(struct board *board, int row) {
    if (!is_board_valid(board) || row < 0 || board->size <= row)
        return NULL;

    return board->arr + row * board->size;
}

inline int *get_col(struct board *board, int col) {
    if (!is_board_valid(board) || col < 0 || board->size <= col)
        return NULL;

    return board->arr + col;
}

inline int get_elem(struct board *board, int row, int col) {
    if (!is_board_valid(board) || !is_index_valid(board, row, col))
        return 0;

    return board->arr[row * board->size + col];
}

inline void set_elem(struct board *board, int row, int col, int value) {
    if (!is_board_valid(board) || !is_index_valid(board, row, col))
        return;

    board->arr[row * board->size + col] = value;
}

void copy_board(struct board *dst, struct board *src) {
    if (dst == NULL || src == NULL)
        return;

    for (int i = 0; i < get_size(src); i++) {
        for (int j = 0; j < get_size(src); j++)
            set_elem(dst, i, j, get_elem(src, i, j));
    }
}

void print_board(struct board *board) {
    print_subboard(board, get_size(board));
}

void print_subboard(struct board *board, int size) {
    if (!is_board_valid(board))
        return;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            printf("%s", get_elem(board, i, j) ? "⬛" : "⬜");
        printf("\n");
    }
}

inline static int is_board_valid(struct board *board) {
    return board != NULL && board->arr != NULL;
}

inline static int is_index_valid(struct board *board, int row, int col) {
    return 0 <= row && row < board->size && 0 <= col && col < board->size;
}