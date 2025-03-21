#ifndef __PROJ1_MATRIX_H__
#define __PROJ1_MATRIX_H__

#include <functional>
#include <string>
#include <vector>

class SquareMatrix
{
    const size_t _size;
    std::vector<double> _matrix;

  public:
    SquareMatrix(size_t size);

    double &operator()(size_t i, size_t j);
    const double &operator()(size_t i, size_t j) const;
    void swap_rows(size_t i, size_t j);
    void swap_ranges(size_t i, size_t j, size_t n);
    size_t size() const;
    std::string to_string(size_t width = 5, size_t precision = 1) const;
    void for_all(std::function<double(size_t, size_t)>);
    double norm() const;
};

SquareMatrix operator+(const SquareMatrix &a, const SquareMatrix &b);
SquareMatrix operator-(const SquareMatrix &a, const SquareMatrix &b);
SquareMatrix operator*(const SquareMatrix &a, const SquareMatrix &b);
SquareMatrix operator*(const SquareMatrix &a, double b);
SquareMatrix operator*(double b, const SquareMatrix &a);
SquareMatrix operator/(const SquareMatrix &a, double b);

#endif