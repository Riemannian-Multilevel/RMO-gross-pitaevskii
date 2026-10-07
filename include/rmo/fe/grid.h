//
// Created by Ferdinand Vanmaele on 12.01.26.
//

#ifndef RMO_FE_GRID_H
#define RMO_FE_GRID_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

/**
 * @file
 * @brief Hypercube domains \f$ [-R, R]^{dim} \f$ with quadrilateral/hexahedral or simplex meshes, and grid
 * output.
 */
namespace rmo::fe
{

/** @brief Triangulation of \f$ [-R, R]^{dim} \f$, globally refined to a number of levels. */
template <int dim>
struct HyperCube
{
    dealii::Triangulation<dim> triangulation;

    /** @brief Simplex (triangles, tetrahedra) instead of quadrilateral/hexahedral mesh. */
    const bool has_simplex = false;

    /**
     * @brief Coarse mesh of \f$ [-R, R]^{dim} \f$ with \f$ R \f$ = @p radius: one cell, or its subdivision
     * into simplices if @p simplex_mesh is set.
     */
    HyperCube(const double radius, bool simplex_mesh) : has_simplex(simplex_mesh)
    {
        simplex_mesh ? simplex(radius) : quadrilateral(radius);
    }

    /** @brief Refines globally to @p n_levels mesh levels, i.e. `n_levels - 1` refinements of the coarse mesh. */
    void refine(unsigned int n_levels)
    {
        // Assert that we are requesting at least 1 level (the coarse mesh itself)
        Assert(n_levels > 0, dealii::ExcMessage("n_levels must be >= 1"));

        if (n_levels > 1) {
            triangulation.refine_global(n_levels - 1); // #cells *= 2^(dim) per step
        }
        AssertDimension(n_levels, triangulation.n_global_levels());
    }

private:
    /** @brief Coarse simplex mesh: the subdivided coarse hypercube. */
    void simplex(double radius)
    {
        dealii::Triangulation<dim> tmp;
        dealii::GridGenerator::hyper_cube(tmp, -radius, radius);

        // uses default number of subdivisions (dim==2 ? 8u : 24u)
        // Note: converting before refinement ensures the coarse grid is simplicial,
        // allowing consistent refinement of simplices.
        dealii::GridGenerator::convert_hypercube_to_simplex_mesh(tmp, triangulation);
    }

    /** @brief Coarse quadrilateral/hexahedral mesh: a single cell. */
    void quadrilateral(double radius)
    {
        dealii::GridGenerator::hyper_cube(triangulation, -radius, radius);
    }
};

/** @brief Writes @p triangulation to @p filename in @p format (e.g. vtk, svg). */
template <int dim>
void write_grid(const std::string& filename, const dealii::Triangulation<dim>& triangulation,
    const dealii::GridOut::OutputFormat format)
{
    std::ofstream out(filename);
    const dealii::GridOut grid_out;

    grid_out.write(triangulation, out, format);
}

/** @brief Writes a 2D @p triangulation to the SVG file @p s. */
inline void
write_grid_svg(const std::string& s, const dealii::Triangulation<2>& triangulation)
{
    write_grid(s, triangulation, dealii::GridOut::OutputFormat::svg);
}

} // namespace rmo::fe
#endif //RMO_FE_GRID_H