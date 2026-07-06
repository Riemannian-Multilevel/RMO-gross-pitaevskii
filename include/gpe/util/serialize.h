#ifndef GPE_SERIALIZE_H
#define GPE_SERIALIZE_H

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/vector.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpe
{

/**
 * @brief Writes the real-space support point coordinates of every DoF to a
 * simple binary file, for post-processing solution vectors outside deal.II
 * (e.g. rasterizing to a fixed-resolution image in Python instead of writing
 * one SVG polygon per cell, which does not scale to large DoF counts).
 *
 * Layout: uint32 magic | uint32 dim | uint64 n_dofs | n_dofs*dim doubles,
 * row-major (x0,y0,[z0,] x1,y1,[z1,] ...).
 *
 * Call this once per mesh/level -- support points are shared by every
 * solution iterate on that level and do not need to be repeated per file.
 */
template <int dim>
void write_support_points(const dealii::DoFHandler<dim>& dof_handler,
                          const dealii::Mapping<dim>& mapping,
                          const std::string& filename)
{
    std::vector<dealii::Point<dim>> points(dof_handler.n_dofs());
    dealii::DoFTools::map_dofs_to_support_points(mapping, dof_handler, points);

    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("write_support_points: could not open " + filename);
    }

    const std::uint32_t magic  = 0x43504547;  // "GEPC"
    const std::uint32_t dim32  = dim;
    const std::uint64_t n_dofs = points.size();

    out.write(reinterpret_cast<const char*>(&magic),  sizeof(magic));
    out.write(reinterpret_cast<const char*>(&dim32),  sizeof(dim32));
    out.write(reinterpret_cast<const char*>(&n_dofs), sizeof(n_dofs));

    for (const auto& p : points) {
        for (unsigned d = 0; d < dim; d++) {
            const double c = p[d];
            out.write(reinterpret_cast<const char*>(&c), sizeof(double));
        }
    }
}

/**
 * @brief Writes a raw solution vector to a simple binary file for
 * post-processing outside deal.II. DoF ordering must match the
 * write_support_points() call for the same level.
 *
 * Layout: uint32 magic | uint64 n_dofs | n_dofs doubles.
 */
inline void write_solution(const dealii::Vector<double>& solution, const std::string& filename)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("write_solution: could not open " + filename);
    }

    const std::uint32_t magic  = 0x53504547;  // "GEPS"
    const std::uint64_t n_dofs = solution.size();

    out.write(reinterpret_cast<const char*>(&magic),  sizeof(magic));
    out.write(reinterpret_cast<const char*>(&n_dofs), sizeof(n_dofs));
    out.write(reinterpret_cast<const char*>(solution.begin()), n_dofs * sizeof(double));
}

} // namespace gpe

#endif //GPE_SERIALIZE_H
