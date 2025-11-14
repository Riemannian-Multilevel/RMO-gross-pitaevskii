#ifndef GPE_MESH_HH
#define GPE_MESH_HH

// step 1 -- grid libraries
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <fstream>

namespace gpe
{
enum class exportFormat
{
    SVG,
    VTK,
    GNUPLOT
};

//!
//! @tparam dim Dimension of the grid
//! @param s Output file name
//! @param triangulation Triangulation<> object containing the grid
//! @param format Output file format
template <int dim>
void grid2file(const std::string& s, const dealii::Triangulation<dim>& triangulation,
               const exportFormat format)
{
    std::ofstream out(s);
    const dealii::GridOut grid_out;

    switch (format) {
    case exportFormat::SVG:
        grid_out.write_svg(triangulation, out);
        break;
    case exportFormat::VTK:
        grid_out.write_vtk(triangulation, out);
        break;
    case exportFormat::GNUPLOT:
        grid_out.write_gnuplot(triangulation, out);
        break;
    default:
        throw std::invalid_argument("unknown format");
    }
    std::cout << "Grid written to " + s << std::endl;
}

//! Write a VTK file for the 2d grid, colored by refinement
//!
//! @tparam dim Dimension of the grid
//! @param s File name to write to, without extension
//! @param triangulation Triangulation<> object containing the grid
inline void grid2svg(const std::string& s, const dealii::Triangulation<2>& triangulation)
{
    grid2file(s, triangulation, exportFormat::SVG);
}

} // namespace gpe

#endif // GPE_MESH_HH
