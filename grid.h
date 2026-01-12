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
class HyperCube
{
public:
    HyperCube(double radius_)
        : triangulation(dealii::Triangulation<dim>::limit_level_difference_at_vertices), radius(radius_)
    {}
    virtual ~HyperCube() = default;

    void setup_grid(unsigned int n_levels)
    {
        // step 1 - regularly refined mesh
        dealii::GridGenerator::hyper_cube(triangulation, -radius, radius);

        // the number of cells increases by a factor of 2^(dim x times)
        triangulation.refine_global(n_levels-1);
        AssertDimension(n_levels, triangulation.n_global_levels());

        std::cerr << "Number of levels: " << triangulation.n_global_levels() << std::endl;
        std::cerr << "Number of vertices: " << triangulation.n_vertices() << std::endl;
    }

    const dealii::Triangulation<dim>& get_triangulation() const {
        return triangulation;
    }

protected:
    dealii::Triangulation<dim> triangulation;
    double radius;
};

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

template <int dim>
void plot_grid(const dealii::Triangulation<dim>& triangulation, const std::string& prefix)
{
    const std::string filename = prefix + "_" + std::to_string(dim) + "d_lvl" + std::to_string(triangulation.n_levels());
    if (dim == 2) {
        grid2file<dim>(filename + ".svg", triangulation, dealii::GridOut::OutputFormat::svg);
    }
    grid2file<dim>(filename + ".gnuplot", triangulation, dealii::GridOut::OutputFormat::gnuplot);
}

//! Write a VTK file for the 2d grid, colored by refinement
//!
//! @tparam dim Dimension of the grid
//! @param s File name to write to, without extension
//! @param triangulation Triangulation<> object containing the grid
inline void
grid2svg(const std::string& s, const dealii::Triangulation<2>& triangulation)
{
    grid2file(s, triangulation, dealii::GridOut::OutputFormat::svg);
}

}
#endif //GPE_GRID_H