#ifndef GPE_UTIL_H
#define GPE_UTIL_H

#include <deal.II/base/point.h>
#include <deal.II/numerics/data_out.h>

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
//! @tparam dim Problem dimension
//! @param solution
//! @param dof_handler
//! @param format
//! @param filename
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

template <int dim>
void output_hdf5(const dealii::Vector<double>& solution, const dealii::DoFHandler<dim>& dof_handler,
    const std::string& filename_h5)
{
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "psi");
    data_out.build_patches(dof_handler.get_fe().degree);

    dealii::DataOutBase::DataOutFilterFlags flags(true, true);
    dealii::DataOutBase::DataOutFilter data_filter(flags);
    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter, filename_h5, MPI_COMM_WORLD);
}

} //namespace gpe

#endif //GPE_UTIL_H