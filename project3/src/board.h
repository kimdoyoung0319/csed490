#ifndef BOARD_H_
#define BOARD_H_

struct board {
    int size;
    int *arr;
};

struct board *make_board(int size);
void destroy_board(struct board *board);

int get_size(struct board *board);
int *get_arr(struct board *board);
int *get_row(struct board *board, int row);
int *get_col(struct board *board, int col);

int get_elem(struct board *board, int row, int col);
void set_elem(struct board *board, int row, int col, int value);

void copy_board(struct board *dst, struct board *src);

void print_board(struct board *board);
void print_subboard(struct board *board, int size);

#endif