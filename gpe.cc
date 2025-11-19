//
// Created by Ferdinand Vanmaele on 01.10.25.
//
#include "mesh.h"
#include "dofs.h"
#include "assemble.h"
#include "lac.h"
#include "util.h"

#include <iostream>
#include <fmt/format.h>

// Formulate possible extensions as command-line options.
#include <boost/program_options.hpp>
#include <boost/algorithm/string/case_conv.hpp>
namespace po = boost::program_options;

using namespace gpe;
using namespace dealii;

template <typename T>
struct GPE_Mass
{
    explicit GPE_Mass(const SparsityPattern &sparsity_)
        // avoid issues with deleted copy/move constructors of SparsityPattern
        // SparseMatrix has defined move constructors, so we store them directly
        : sparsity(std::make_shared<dealii::SparsityPattern>())
    {
        // make an internal copy of the sparsity pattern
        sparsity->copy_from(sparsity_);

        // now initialize all matrices with this *owned* pattern
        M.reinit(*sparsity);
        Mv.reinit(*sparsity);
        S.reinit(*sparsity);
        Mpp.reinit(*sparsity);
    }

    SparseMatrix<T> M;
    SparseMatrix<T> Mv;
    SparseMatrix<T> S;
    SparseMatrix<T> Mpp;

private:
    std::shared_ptr<SparsityPattern> sparsity;
};

//!
//! @tparam dim
template <int dim>
class GPE
{
public:
    //!
    //! @param n_levels_ Number of times the mesh is refined
    //! @param degree Degree of the Lagrange finite element
    //! @param left_ Left end-point of the rectangular domain
    //! @param right_ Opposite end-point of the rectangular domain
    //! @param order_ Ordering used for degrees of freedom
    GPE(const int n_levels_, const int degree, const Point<dim>& left_, const Point<dim>& right_,
        const Ordering order_ = Ordering::DEFAULT)
    :
        n_levels(n_levels_), order(order_), left(left_), right(right_),
        // Flag to allow multigrid algorithms
        triangulation(Triangulation<dim>::limit_level_difference_at_vertices),
        // DoFHandler<> has a deleted assignment operator, so initialize in the constructor
        element(degree), dof_handler(triangulation)
    {
        dimension = dim;
    }

    void make_rectangle()
    {
        // step 1 - make grid
        // rectangle consisting of precisely one cell
        std::cerr << "Dimension: " << dimension << std::endl;
        dealii::GridGenerator::hyper_rectangle(triangulation, left, right);

        // the number of cells increases by a factor of 2^(dim x times)
        triangulation.refine_global(n_levels);
        std::cerr << "Number of cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of levels: " << triangulation.n_levels() << std::endl;
    }

    void dofs()
    {
        // step 2 - degrees of freedom
        distribute_dofs(dof_handler, element, order);
    }

    void dofs_mg()
    {
        // step 2 - degrees of freedom - ordering applied to every level
        std::vector<int> levels(n_levels);
        std::iota(levels.begin(), levels.end(), 1);

        // DoFHandler::distribute_dofs, DoFHandler::distribute_mg_dofs
        distribute_mg_dofs(dof_handler, element, order, levels);
    }

    template <typename Function>
    // Use vector with one entry for interoperability with assemble_mg()
    std::vector<GPE_Mass<double> >
    assemble(Function&& V) const
    {
        std::vector<GPE_Mass<double> > Mass_v;
        // In-place construction of sparsity pattern
        Mass_v.emplace_back(make_sparsity_pattern(dof_handler));
        std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        // Step 3 - linear system
        // Compute values of mass matrix
        assemble_mass(Mass_v[0].M, dof_handler);
        assemble_mass_weighed(Mass_v[0].Mv, dof_handler, V);

        // Compute values of stiffness matrix
        assemble_stiffness(Mass_v[0].S, dof_handler);
        return Mass_v; // XXX: or store in class object
    }

    template <typename Function>
    std::vector<GPE_Mass<double> >
    assemble_mg(Function&& V) const
    {
        // Structure of arrays for multigrid
        //std::vector<SparseMatrix<double> > mass_matrix_weighed_v(n_levels);
        //std::vector<SparseMatrix<double> > stiffness_matrix_v(n_levels);
        std::vector<GPE_Mass<double> > Mass_v;
        std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        // Iterate over cells in every level of the hierarchy
        for (int i = 0; i < n_levels; i++) {
            Mass_v.emplace_back(make_sparsity_pattern_mg(i, dof_handler));
        }
        // Compute stiffness and weighed mass matrices
        for (int i = 0; i < n_levels; i++) {
            // compute values of mass matrix for level i
            assemble_mass(Mass_v[i].M, dof_handler);
            assemble_mass_weighed(Mass_v[i].Mv, dof_handler, V);

            // compute values of stiffness matrix for level i
            assemble_stiffness(Mass_v[i].S, dof_handler);
        }
        return Mass_v; // XXX: or store in class (mg specialized?) object
    }

    const DoFHandler<dim>& get_dof() const {
        return dof_handler;
    }

private:
    // Problem parameters
    int n_levels;
    int dimension;
    Ordering order;

    // Rectangle bounds
    Point<dim> left;
    Point<dim> right;

    // Finite element containers
    Triangulation<dim>   triangulation; // copy stored by dof_handler
    const FE_Q<dim>      element;       // copy stored by dof_handler
    DoFHandler<dim>      dof_handler;
};

//!
//! @param M Sparse matrix
//! @return Copy of input sparse matrix
SparseMatrix<double>
sp_copy(const SparseMatrix<double>& M)
{
    SparseMatrix<double> M_copy;
    M_copy.reinit(M);
    M_copy.copy_from(M);
    return M_copy;
}

//! Riemannian gradient descent for the GPE energy minimization
//! @tparam dim Problem dimension
//! @param Mass Finite element matrix object (mass, stiffness, weighed mass)
//! @param dof_handler DoF object for assembling weighed mass matrices
//! @param x0 Starting value
//! @param beta Non-linearity factor for GPE
//! @param h Step-size for Riemannian gradient descent (RGD)
//! @param solver Used sparse solver (gmres|minres|cg)
//! @param max_iter Maximum number of RGD iterations
//! @param max_iter_inner Maximum number of sparse solver iterations
//! @param reltol Relative tolerance for sparse solver
//! @return
template <int dim>
std::tuple<Vector<double>,double,double>
// TODO: move gradient methods to separate header
rgd_fixed_step(GPE_Mass<double>& Mass, const dealii::DoFHandler<dim>& dof_handler,
    const Vector<double>& x0, double beta, double h,
    SolverMethod solver, int max_iter, int max_iter_inner, double reltol)
{
    // TODO: switch to gsl-lite or deal-ii assertions
    assert(h > 0);
    assert(reltol > 0);

    // Generate sparsity pattern, weighed mass matrix, and stiffness matrix
    SparseMatrix<double> A_0 = sp_copy(Mass.S);
    A_0.add(1.0, Mass.Mv);

    // Begin RGD iteration
    Vector<double> x(x0);
    for (int i = 0; i < max_iter; i++) {
        SparseMatrix<double> A = sp_copy(A_0);
        assemble_mass_phiphi(Mass.Mpp, dof_handler,  x);
        A.add(beta, Mass.Mpp);  // A = A_0 + beta * M_phiphi

        // Invert A
        Vector<double> y = solve_sparse(A, x, solver, max_iter_inner, reltol);
        double denom1 = x * y; // x' A^-1 x
        assert(denom1 > 0);

        Vector<double> z(y);
        z /= denom1; // A^-1 x / (x' A^-1 x)
        z.add(-1.0, x); // x - z

        // Normalize in energy norm
        x.add(-h, z); // x - h(x - z)
        Vector<double> tmp(x.size());
        Mass.M.vmult(tmp, x);
        double denom2 = std::sqrt(x * tmp);
        x /= denom2; // (x - h(x - z)) / ||x - h(x-z)||_M
    }

    // Compute residual
    SparseMatrix<double> A = sp_copy(A_0);
    assemble_mass_phiphi<dim>(Mass.Mpp, dof_handler, x);

    Vector<double> res(x.size());
    A.add(beta, Mass.Mpp);
    A.vmult(res, x); // A*x

    Vector<double> tmp(x);
    tmp *= (x*res); // (x'*A*x)*x
    res.add(-1.0, tmp); // A*x - (x'*A*x)*x

    // Compute energy
    SparseMatrix<double> B = sp_copy(A_0);
    B *= 0.5;
    B.add(0.25, Mass.Mpp);

    Vector<double> tmp1(x.size());
    B.vmult(tmp1, x);
    double energy = x * tmp1;

    // TODO: residual in the energy norm
    return std::make_tuple(x, res.l2_norm() / tmp.l2_norm(), energy);
}

template <int dim>
void experiment1(int n_levels, int degree,
    const std::string& left_str, const std::string& right_str,
    double beta, double h, Ordering order, bool multigrid,
    SolverMethod solver, int max_iter, int max_iter_inner, double reltol)
{
    // Define potential function
    auto V = [](const Point<dim>& p) {
        typename Point<dim>::value_type out = 0.0;
        for (unsigned d = 0; d < dim; d++) {
            out += p[d]*p[d];
        }
        return out;
    };

    // Set up grid
    GPE<dim> Problem(n_levels, degree, str_to_point<dim>(left_str), str_to_point<dim>(right_str), order);
    Problem.make_rectangle();

    // Set up mass and stiffness matrices
    std::vector<GPE_Mass<double> > Mass_v;
    if (multigrid) {
        Problem.dofs_mg();
        Mass_v = Problem.assemble_mg(V);
    } else {
        Problem.dofs();
        Mass_v = Problem.assemble(V);
    }

    // Run iteration, update M_phiphi in every step
    const DoFHandler<dim>& dof_handler = Problem.get_dof();
    std::tuple<Vector<double>, double, double> results;

    if (multigrid) {
        // TODO: run iteration for all levels
        //       initial value for each level in the multigrid hierarchy - n_dofs(level)
        throw std::logic_error("inverse_iteration_mg not implemented");
    }
    else {
        // Set initial value
        Vector<double> x0(Problem.get_dof().n_dofs());
        x0 = 1.0;

        results = rgd_fixed_step<dim>(Mass_v[0], dof_handler, x0, beta, h,
            solver, max_iter, max_iter_inner, reltol);
    }
    std::cerr << "Residual: " << std::get<1>(results) << std::endl;
    std::cerr << "Energy: " << std::get<2>(results) << std::endl;
}

int main(int argc, char* argv[])
{
    int degree, n_levels, dimension, max_iter, max_iter_inner;
    bool multigrid;
    Ordering order;
    std::string left_str, right_str;
    SolverMethod solver;
    double beta, reltol, step_size;

    // TODO: add configuration file (cf. boost tutorial)
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("degree", po::value<int>()->default_value(1),
             "polynomial degree for finite element")
            ("levels", po::value<int>()->default_value(2),
             "number of times to globally refine the mesh")
            ("multigrid", po::value<bool>()->default_value(false),
             "enable multigrid")
            ("dimension", po::value<int>()->default_value(2),
             "problem dimension")
            ("ordering", po::value<std::string>()->default_value("default"),
             "ordering for degrees of freedom (default|random|cuthill_mckee|king|min_deg)")
            ("left", po::value<std::string>()->default_value(""),
                "left point of the mesh")
            ("right", po::value<std::string>()->default_value(""),
                "right point of the mesh")
            ("beta", po::value<double>()->default_value(100),
                "non-linearity factor")
            ("solver", po::value<std::string>()->default_value("gmres"),
                "sparse solver (gmres|minres|cg)")
            ("max_iter", po::value<int>()->default_value(25),
                "maximum number of iterations")
            ("max_iter_inner", po::value<int>()->default_value(100),
                "maximum number of iterations for sparse solver")
            ("reltol", po::value<double>()->default_value(1e-6),
                "relative tolerance for sparse solver")
            ("step_size", po::value<double>()->default_value(1.0),
                "step size for RGD");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        auto order_str = vm["ordering"].as<std::string>();
        auto solver_str = vm["solver"].as<std::string>();
        std::transform(order_str.begin(), order_str.end(), order_str.begin(), ::toupper);
        std::transform(solver_str.begin(), solver_str.end(), solver_str.begin(), ::toupper);

        order     = select_order(order_str);
        degree    = vm["degree"].as<int>();
        n_levels  = vm["levels"].as<int>();
        dimension = vm["dimension"].as<int>();
        multigrid = vm["multigrid"].as<bool>();
        left_str  = vm["left"].as<std::string>();
        right_str = vm["right"].as<std::string>();
        beta      = vm["beta"].as<double>();
        solver    = select_solver(solver_str);
        max_iter  = vm["max_iter"].as<int>();
        max_iter_inner = vm["max_iter_inner"].as<int>();
        reltol    = vm["reltol"].as<double>();
        step_size = vm["step_size"].as<double>();
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "Exception of unknown type!\n";
        return 1;
    }

    switch (dimension) {
    case 1:
        {
            // Default endpoints of rectangle
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10"; right_str = "10";
            }
            experiment1<1>(n_levels, degree, left_str, right_str, beta, step_size,
                           order, multigrid, solver, max_iter, max_iter_inner, reltol);
        }
        break;
    case 2:
        {
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10"; right_str = "10,10";
            }
            experiment1<2>(n_levels, degree, left_str, right_str, beta, step_size,
                           order, multigrid, solver, max_iter, max_iter_inner, reltol);
        }
        break;
    case 3:
        {
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10,-10"; right_str = "10,10,10";
            }
            experiment1<3>(n_levels, degree, left_str, right_str, beta, step_size,
                           order, multigrid, solver, max_iter, max_iter_inner, reltol);
        }
        break;
    default:
        throw std::invalid_argument("dimension must be 1, 2 or 3");
    }
}
