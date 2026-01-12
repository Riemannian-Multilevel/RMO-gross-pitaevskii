#ifndef GPE_UTIL_H
#define GPE_UTIL_H

#include <deal.II/base/point.h>
#include <deal.II/grid/grid_out.h>

#include <type_traits>

namespace gpe
{

// Trick to distinguish between multigrid and regular solver packages at compile-time
struct mg_solver_tag {};
struct plain_solver_tag {};

template <class Solver, class Tag>
inline constexpr bool is_solver_kind_v = std::is_same_v<typename Solver::solver_kind, Tag>;


// Utility function for taking boundary points as strings "x,y,z" from the command-line
template <int dim>
dealii::Point<dim> str_to_point(const std::string& s, const char sep=',') {
    dealii::Point<dim> p;
    std::stringstream ss(s);
    std::string item;

    int i = 0;
    while (std::getline(ss, item, sep) && i < dim) {
        p[i++] = std::stod(item);
    }
    assert(i == dim);
    return p;
}

// Utility function for selecting dimension (compile-time) at runtime
template <class F>
decltype(auto) with_dimension(unsigned dim, F&& f)
{
    switch (dim)
    {
        case 1: return std::forward<F>(f)(std::integral_constant<int, 1>{});
        case 2: return std::forward<F>(f)(std::integral_constant<int, 2>{});
        case 3: return std::forward<F>(f)(std::integral_constant<int, 3>{});
        default:
            throw std::invalid_argument("dimension must be 1, 2 or 3");
    }
}

//!
//! @tparam dim Dimension of the grid
//! @param filename Output file name
//! @param triangulation Triangulation<> object containing the grid
//! @param format Output file format
template <int dim>
void grid2file(const std::string& filename, const dealii::Triangulation<dim>& triangulation,
    const dealii::GridOut::OutputFormat format)
{
    std::ofstream out(filename);
    const dealii::GridOut grid_out;

    grid_out.write(triangulation, out, format);
    std::cout << "Grid written to " + filename << std::endl;
}

//! Write a VTK file for the 2d grid, colored by refinement
//!
//! @tparam dim Dimension of the grid
//! @param s File name to write to, without extension
//! @param triangulation Triangulation<> object containing the grid
inline void grid2svg(const std::string& s, const dealii::Triangulation<2>& triangulation)
{
    grid2file(s, triangulation, dealii::GridOut::OutputFormat::svg);
}

} //namespace gpe

#endif //GPE_UTIL_H