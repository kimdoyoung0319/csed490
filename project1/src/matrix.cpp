#include "matrix.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

SquareMatrix::SquareMatrix(size_t size) : _size(size), _matrix(size * size) {};

double &SquareMatrix::operator()(size_t i, size_t j)
{
    return _matrix[i * _size + j];
};

const double &SquareMatrix::operator()(size_t i, size_t j) const
{
    return _matrix[i * _size + j];
}

void SquareMatrix::swap_rows(size_t i, size_t j) { swap_ranges(i, j, _size); }

void SquareMatrix::swap_ranges(size_t i, size_t j, size_t n)
{
    if (i >= _size || j >= _size || n >= _size)
        return;

    std::swap_ranges(_matrix.begin() + i * _size,
                     _matrix.begin() + i * _size + n,
                     _matrix.begin() + j * _size);
}

size_t SquareMatrix::size() const { return _size; }

std::string SquareMatrix::to_string(size_t width, size_t precision) const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision);

    for (size_t i = 0; i < _size; i++)
    {
        for (size_t j = 0; j < _size; j++)
        {
            oss << std::setw(width) << _matrix[i * _size + j] << ' ';
        }

        oss << '\n';
    }

    return oss.str();
}