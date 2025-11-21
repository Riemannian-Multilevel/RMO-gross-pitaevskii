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

} // namespace gpe

#endif // GPE_MESH_HH
