#ifndef RMO_UTIL_SERIALIZE_H
#define RMO_UTIL_SERIALIZE_H

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

/**
 * @file
 * @brief Raw binary output of DoF coordinates and solution vectors, for post-processing outside deal.II
 * (e.g. rasterizing in Python, see staging/plot_solution.py), which scales to large numbers of DoFs.
 */
namespace rmo
{

//! Writes @p solution (named "psi") to @p filename in @p format.
template <int dim>
void output_results(const dealii::Vector<double>& solution, const dealii::DoFHandler<dim>& dof_handler,
    const dealii::DataOutBase::OutputFormat format, const std::string& filename)
{
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "psi");
    data_out.build_patches(dof_handler.get_fe().degree);

    std::ofstream output(filename);
    data_out.write(output, format);
}

//! Writes the final iterate to `basename.vtk` and, if @p every > 0, also every k-th and the final iterate to
//! `basename_iter<i>.vtk`, which ParaView reads as one time series.
template <int dim>
void output_vtk(const std::vector<dealii::Vector<double>>& history, const dealii::DoFHandler<dim>& dof_handler,
                const std::string& basename, unsigned every)
{
    AssertThrow(!history.empty(), dealii::ExcMessage("no iterates to write"));

    output_results(history.back(), dof_handler, dealii::DataOutBase::vtk, basename + ".vtk");
    if (every == 0) {
        return;
    }

    const std::size_t last = history.size() - 1;
    for (std::size_t i = 0; i <= last; ++i) {
        if (i == last || i % every == 0) {
            output_results(history[i], dof_handler, dealii::DataOutBase::vtk,
                           basename + "_iter" + std::to_string(i) + ".vtk");
        }
    }
}

/**
 * @brief Writes the support points of all DoFs; once per mesh, shared by all solution files on it.
 *
 * Layout: uint32 magic | uint32 dim | uint64 n_dofs | n_dofs*dim doubles, row-major (x0, y0, [z0,] x1, ...).
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
 * @brief Writes a solution vector, in the DoF order of write_support_points() for the same mesh.
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

/**
 * @brief Writes a sparse matrix to a file in the MatrixMarket (coordinate real general) format.
 *
 * @param A `SparseMatrix` to be serialized.
 * @param filename Output file name.
 */
inline void write_matrix_market(const dealii::SparseMatrix<double>& A, const std::string& filename)
{
    std::ofstream out(filename + ".mtx");
    AssertThrow(out, dealii::ExcMessage("could not open " + filename));

    out << "%%MatrixMarket matrix coordinate real general\n"
        << A.m() << " " << A.n() << " " << A.n_nonzero_elements() << "\n"
        << std::setprecision(16);

    // Iterate over the row indices, and then access each row by column-index. This approach is recommended
    // for performance; see the documentation of SparseMatrixIterators::Iterator.
    for (unsigned i = 0; i < A.m(); ++i) {
        for (auto it = A.begin(i); it != A.end(i); ++it) {
            out << i + 1 << " " << it->column() + 1 << " " << it->value() << "\n";
        }
    }
}

inline void write_matrix_market(const dealii::DiagonalMatrix<Vector<double>>& A, const std::string& filename)
{
    std::ofstream out(filename + ".mtx");
    AssertThrow(out, dealii::ExcMessage("could not open " + filename));

    out << "%%MatrixMarket matrix coordinate real general\n"
        << A.m() << " " << A.n() << " " << A.get_vector().size() << "\n"
        << std::setprecision(16);

    for (unsigned i = 0; i < A.m(); ++i) {
        out << i + 1 << " " << i + 1 << " " << A(i, i) << "\n";
    }
}

} // namespace rmo

#endif //RMO_UTIL_SERIALIZE_H
