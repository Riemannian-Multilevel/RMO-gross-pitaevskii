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

//! Generate cube domain with global refinement of cells
//! @tparam dim
//! @param triangulation
//! @param radius
//! @param n_levels
template <int dim>
void make_cube(dealii::Triangulation<dim>& triangulation, double radius, int n_levels)
{
    dealii::GridGenerator::hyper_cube(triangulation, -radius, radius);

    // the number of cells increases by a factor of 2^(dim x times)
    triangulation.refine_global(n_levels-1);
    AssertDimension(n_levels, triangulation.n_global_levels());
}

//! Generate cube domain with refinement of cells based on distance to the origin
//! @tparam dim Problem dimension
//! @param triangulation Triangulation object to be initialized
//! @param radius Radius of the cube domain
//! @param R_outer Start refining inside this radius (defaults to radius*sqrt(dim))
//! @param R_inner Target radius for finest cells (defaults to 0.5)
//! @param n_levels_graded Number of adaptive refinements
//! @param n_levels Number of global refinements
template <int dim>
void make_cube_graded(dealii::Triangulation<dim>& triangulation, double radius,
    const int n_levels = 3, const int n_levels_graded = 2,
    double R_outer = -1.0, double R_inner = -1.0)
{
    // defaults for controlling grading strength
    if (R_outer == -1.0) { R_outer = radius*(static_cast<double>(dim)); }
    if (R_inner == -1.0) { R_inner = 0.5; }

    // Geometric shrink factor for the refinement radius per cycle
    const double shrink = std::pow(R_inner / R_outer, 1.0 / n_levels_graded);

    // Start from a very coarse mesh on [-L, L]^dim
    dealii::GridGenerator::hyper_cube(triangulation, -radius, radius);
    triangulation.refine_global(n_levels);

    double current_radius = R_outer;

    for (int cycle = 0; cycle < n_levels_graded; ++cycle)
    {
        // Mark cells for refinement if their center is closer than current_radius to the origin
        for (const auto &cell : triangulation.active_cell_iterators())
        {
            if (const double r = cell->center().norm(); r < current_radius) {
                cell->set_refine_flag();
            }
        }
        triangulation.execute_coarsening_and_refinement();
        current_radius *= shrink;
    }
}

// TODO: adaptive grid based on KellyErrorEstimate (tutorial/step-6)

//!
//! @tparam dim Dimension of the grid
//! @param filename Output file name
//! @param triangulation Triangulation<> object containing the grid
//! @param format Output file format
template <int dim>
void write_grid(const std::string& filename, const dealii::Triangulation<dim>& triangulation,
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
    write_grid(s, triangulation, dealii::GridOut::OutputFormat::svg);
}

} // namespace gpe

#endif // GPE_MESH_HH
