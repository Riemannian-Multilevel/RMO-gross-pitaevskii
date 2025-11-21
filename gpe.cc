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

// TODO: only store M, A_0 and Mpp
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
        // -> n_levels equals the number of refinements + 1
        triangulation.refine_global(n_levels-1);

        AssertDimension(n_levels, triangulation.n_levels());
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

        // Iterate over cells in every level of the hierarchy
        for (int level = 0; level < n_levels; level++) {
            Mass_v.emplace_back(make_sparsity_pattern_mg(level, dof_handler));
        }
        // Compute stiffness and weighed mass matrices
        for (int level = 0; level < n_levels; level++) {
            // compute values of mass matrix for level
            assemble_mass(Mass_v[level].M, dof_handler, level);
            assemble_mass_weighed(Mass_v[level].Mv, dof_handler, V, level);

            // compute values of stiffness matrix for level
            assemble_stiffness(Mass_v[level].S, dof_handler, level);
        }
        return Mass_v; // XXX: or store in class (mg specialized?) object
    }

    const DoFHandler<dim>& get_dof() const
    {
        return dof_handler;
    }
    const Triangulation<dim>& get_triangulation() const
    {
        return triangulation;
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

struct GPE_Control
{
    double mass;
    double lambda;
    double residual;
    double rg_norm;
};

template <typename Matrix>
void energy_residual(GPE_Control& control, const Vector<double>& x, const Vector<double>& g,
                     const Matrix& A, const Matrix& M)
{
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    control.mass = x * Mx;      // should be ~ 1 (energy constraint)

    Vector<double> Ax(x.size());
    A.vmult(Ax, x);
    control.lambda = x * Ax;   // Rayleigh quotient (x'Ax / x'Mx)

    Vector<double> r(Ax);
    r.add(-control.lambda, Mx);        // r = A x - lambda M x
    //double res = r.l2_norm(); // or M-norm, see below

    Vector<double> Mr(r.size());
    M.vmult(Mr, r);
    control.residual = std::sqrt(r * Mr);

    Vector<double> Mg(g.size());
    M.vmult(Mg, g);
    control.rg_norm = std::sqrt(g * Mg);

    std::cerr << "Mass = " << control.mass << ", lambda = " << control.lambda << ", residual = " << control.residual
        << std::endl;
}

template <typename Matrix>
[[maybe_unused]] double
energy(const Vector<double>& x, const Matrix& A_0, const Matrix& Mpp)
{
    // Compute energy (factor out?)
    SparseMatrix<double> B = sp_copy(A_0);
    B *= 0.5;
    B.add(0.25, Mpp);

    Vector<double> Bx(x.size());
    B.vmult(Bx, x);
    double E = x * Bx;

    std::cerr << "Energy = " << E << std::endl;
    return E;
}

struct GdOptions
{
    double tol_rel;  // relative tolerance for outer loop (residual)
    double tol_lmb;  // tolerance for rayleigh quotients
    double tol_eig;  // tolerance for M-residual
    int max_iter;    // maximum GD iterations
    int max_inner;   // maximum sparse solver iterations
};

//! Riemannian gradient descent for the GPE energy minimization
//! @tparam dim Problem dimension
//! @param Mass Finite element matrix object (mass, stiffness, weighed mass)
//! @param dof_handler DoF object for assembling weighed mass matrices
//! @param x0 Starting value
//! @param beta Non-linearity factor for GPE
//! @param h Step-size for Riemannian gradient descent (RGD)
//! @param solver Used sparse solver (gmres|minres|cg)
//! @param options Termination criteria
//! @return
template <int dim>
Vector<double>
// TODO: move gradient/energy methods to separate header
//       SolverOptions for inner solve
energy_rgd(GPE_Mass<double>& Mass, const dealii::DoFHandler<dim>& dof_handler,
           const Vector<double>& x0, double beta, double h, SolverMethod solver,
           const GdOptions& options, int check_every = 5, unsigned int level = invalid_unsigned_int)
{
    // TODO: switch to gsl-lite or deal-ii assertions
    assert(h > 0);
    assert(reltol > 0);

    // Generate sparsity pattern, weighed mass matrix, and stiffness matrix
    SparseMatrix<double> A_0 = sp_copy(Mass.S);
    A_0.add(1.0, Mass.Mv);

    // Begin RGD iteration
    Vector x(x0);
    GPE_Control control{};  // initialize fields to 0
    PreconditionIdentity precondition{};

    // TODO: unnecessary copies g, z
    for (int it = 0; it < options.max_iter; it++) {
        // A = A_0 + beta * M_phiphi
        SparseMatrix<double> A = sp_copy(A_0);
        assemble_mass_phiphi<dim>(Mass.Mpp, dof_handler,  x, level);
        A.add(beta, Mass.Mpp);

        // Solve linear system
        Vector<double> y = solve_sparse(A, x, solver,
            precondition, options.max_inner, options.tol_rel);

        // z <- A^{-1}x / (x' A^{-1}x)
        Vector<double> z(y);
        double denom1 = x * y; // x' A^-1 x
        Assert(denom1 > 0, "x' A^{-1} x <= 0");
        z /= denom1;

        // g <- x - z
        Vector<double> g(x);
        g.add(-1.0, z);

        // x <- x - h g
        x.add(-h, g);

        // x <- x / ||x||_M
        Vector<double> Mx(x.size());
        Mass.M.vmult(Mx, x);
        x /= std::sqrt(x * Mx);

        // Check termination criteria every N steps
        // TODO: check M-norm residual and rayleigh quotient (control.lambda)
        if (it % check_every == 0) {
            GPE_Control control_prev = control;
            energy_residual(control, x, g, A, Mass.M);
        }
    }
    return x;
}

template <int dim>
void experiment1(int n_levels, int degree,
    const std::string& left_str, const std::string& right_str,
    double beta, double h, Ordering order, bool multigrid,
    SolverMethod solver, const GdOptions& options)
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
    const Triangulation<dim>& triangulation = Problem.get_triangulation();

    if (multigrid) {
        for (int level = 0; level < n_levels; level++) {
            std::cerr << "Level: " << level << std::endl;
            std::cerr << "Number of cells: " << triangulation.n_cells(level) << std::endl;
            std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs(level) << std::endl;

            Vector<double> x0(Problem.get_dof().n_dofs(level));
            x0 = 1.0;

            Vector<double> x = energy_rgd<dim>(Mass_v[level], dof_handler, x0, beta,
                h, solver, options, 5, level);
            std::cerr << std::endl;
        }
    }
    else {
        std::cerr << "Number of cells: " << triangulation.n_active_cells() << std::endl;
        std::cerr << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        // Set initial value
        Vector<double> x0(Problem.get_dof().n_dofs());
        x0 = 1.0;

        Vector<double> x = energy_rgd<dim>(Mass_v[0], dof_handler, x0, beta,
            h, solver, options, 5);
    }
}

int main(int argc, char* argv[])
{
    int degree, n_levels, dimension;
    bool multigrid;
    Ordering order;
    std::string left_str, right_str;
    SolverMethod solver;
    double beta, step_size;
    GdOptions options{};

    // TODO: add configuration file (cf. boost tutorial)
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("degree", po::value<int>()->default_value(1),
             "polynomial degree for finite element")
            ("levels", po::value<int>()->default_value(3),
             "number of times to globally refine the mesh")
            ("multigrid", po::bool_switch(&multigrid),
             "enable multigrid")
            ("dimension", po::value<int>()->default_value(2),
             "problem dimension")
            ("ordering", po::value<std::string>()->default_value("default"),
             "ordering for degrees of freedom (default|random|cuthill_mckee|king|min_deg)")
            ("left", po::value<std::string>()->default_value(""),
                "left point of the mesh")
            ("right", po::value<std::string>()->default_value(""),
                "right point of the mesh")
            ("beta", po::value<double>()->default_value(100.0),
                "non-linearity factor")
            ("solver", po::value<std::string>()->default_value("gmres"),
                "sparse solver (gmres|minres|cg)")
            ("max_iter", po::value<int>()->default_value(25),
                "maximum number of iterations")
            ("max_inner", po::value<int>()->default_value(100),
                "maximum number of iterations for sparse solver")
            ("eigtol", po::value<double>()->default_value(1e-4),
                "relative tolerance for M-residual")
            ("reltol", po::value<double>()->default_value(1e-6),
                "relative tolerance for sparse solver")
            ("lmbtol", po::value<double>()->default_value(1e-8),
                "relative tolerance for rayleigh quotient")
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
        step_size = vm["step_size"].as<double>();
        options.max_iter  = vm["max_iter"].as<int>();
        options.max_inner = vm["max_inner"].as<int>();
        options.tol_rel   = vm["reltol"].as<double>();
        options.tol_eig   = vm["eigtol"].as<double>();
        options.tol_lmb   = vm["lmbtol"].as<double>();
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
                           order, multigrid, solver, options);
        }
        break;
    case 2:
        {
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10"; right_str = "10,10";
            }
            experiment1<2>(n_levels, degree, left_str, right_str, beta, step_size,
                           order, multigrid, solver, options);
        }
        break;
    case 3:
        {
            // TODO: encode default arguments in function template? (if constexpr...)
            if (left_str.empty() || right_str.empty()) {
                left_str = "-10,-10,-10"; right_str = "10,10,10";
            }
            experiment1<3>(n_levels, degree, left_str, right_str, beta, step_size,
                           order, multigrid, solver, options);
        }
        break;
    default:
        throw std::invalid_argument("dimension must be 1, 2 or 3");
    }
}
