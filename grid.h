//
// Created by Ferdinand Vanmaele on 12.01.26.
//

#ifndef GPE_GRID_H
#define GPE_GRID_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

namespace gpe
{

template <int dim>
struct HyperCube
{
    dealii::Triangulation<dim> triangulation;
    const bool has_simplex = false;

    // For extension with other cell shapes, use MeshKind{} (option_types.h)
    // TODO: expose number of subdivisions
    HyperCube(const double radius, bool simplex_mesh) : has_simplex(simplex_mesh)
    {
        simplex_mesh ? simplex(radius) : quadrilateral(radius);
    }

    void refine(unsigned int n_levels)
    {
        triangulation.refine_global(n_levels-1);   // #cells *= 2^*(dim x times)
        AssertDimension(n_levels, triangulation.n_global_levels());
    }

private:
    void simplex(double radius)
    {
        dealii::Triangulation<dim> tmp;
        dealii::GridGenerator::hyper_cube(tmp, -radius, radius);

        // uses default number of subdivisions (dm==2 ? 8u : 24u)
        // TODO: is it better to subdivide after, or before refinement? (possible distinction on dim 2, 3)
        dealii::GridGenerator::convert_hypercube_to_simplex_mesh(tmp, triangulation);
    }
    void quadrilateral(double radius)
    {
        dealii::GridGenerator::hyper_cube(triangulation, -radius, radius);
    }
};

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
}

//! Write a VTK file for the 2d grid, colored by refinement
//!
//! @tparam dim Dimension of the grid
//! @param s File name to write to, without extension
//! @param triangulation Triangulation<> object containing the grid
inline void
write_grid_svg(const std::string& s, const dealii::Triangulation<2>& triangulation)
{
    write_grid(s, triangulation, dealii::GridOut::OutputFormat::svg);
}

}
#endif //GPE_GRID_H