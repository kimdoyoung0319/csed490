#include "matrix.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

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
    if (i >= _size || j >= _size || n > _size || i == j)
    {
        return;
    }

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

void SquareMatrix::for_all(std::function<double(size_t, size_t)> func)
{
    for (size_t i = 0; i < _size; i++)
    {
        for (size_t j = 0; j < _size; j++)
        {
            _matrix[i * _size + j] = func(i, j);
        }
    }
}

double SquareMatrix::norm() const
{
    double result = 0;

    for (auto it = _matrix.begin(); it != _matrix.end(); it++)
    {
        result += (*it) * (*it);
    }

    return result;
}

SquareMatrix operator+(const SquareMatrix &a, const SquareMatrix &b)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument("the size of two matrices are different");
    }

    SquareMatrix result(a.size());
    result.for_all([&](size_t i, size_t j) { return a(i, j) + b(i, j); });

    return result;
}

SquareMatrix operator-(const SquareMatrix &a, const SquareMatrix &b)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument("the size of two matrices are different");
    }

    SquareMatrix result(a.size());
    result.for_all([&](size_t i, size_t j) { return a(i, j) - b(i, j); });

    return result;
}

SquareMatrix operator*(const SquareMatrix &a, const SquareMatrix &b)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument("the size of two matrices are different");
    }

    SquareMatrix result(a.size());
    result.for_all(
        [&](size_t i, size_t j)
        {
            double r = 0;

            for (size_t k = 0; k < a.size(); k++)
            {
                r += a(i, k) * b(k, j);
            }

            return r;
        });

    return result;
}

SquareMatrix operator*(const SquareMatrix &a, double b)
{
    SquareMatrix result(a.size());
    result.for_all([&](size_t i, size_t j) { return b * a(i, j); });

    return result;
}

SquareMatrix operator*(double a, const SquareMatrix &b) { return b * a; }

//! \todo Is this equivalence holds?
SquareMatrix operator/(const SquareMatrix &a, double b) { return a * (1 / b); }