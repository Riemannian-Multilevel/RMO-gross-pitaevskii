//
// Created by Ferdinand Vanmaele on 12.01.26.
//

#ifndef RMO_OPTION_TYPES_H
#define RMO_OPTION_TYPES_H

#include <optional>
#include <string>
#include <vector>
#include <cstdint>

/**
 * @file
 * @brief Option types of the solvers and the Gross-Pitaevskii problem: enums and option structs, filled from the
 * command line by option.h.
 */
namespace rmo
{

// TODO: consistent naming of classes
// Every ordering should be compatible to (geometric) multigrid
enum class Ordering : std::uint8_t
{
    DEFAULT,
    RANDOM,
    CUTHILL_MCKEE,
    KING,
    MIN_DEG
};

enum class BoundaryCondition : std::uint8_t
{
    NEUMANN,
    DIRICHLET
};

enum class MeshKind : std::uint8_t
{
    QUADRILATERAL,
    SIMPLEX
};

enum class SolverMethod : std::uint8_t
{
    GMRES,
    MINRES,
    CG
};

enum class Precondition : std::uint8_t
{
    NONE,
    DIAGONAL,
    SPARSE_ILU,
    AMG
};

enum class MetricKind : std::uint8_t
{
    NONE,   // Galerkin condition not fulfilled
    FROBENIUS,
    MASS,
    ENERGY_ADAPTIVE
};

enum class ManifoldKind : std::uint8_t
{
    MASS_UNIT_SPHERE,
    MASS_UNIT_SPHERE_POSITIVE,
    SIMPLEX, // TODO
    POSITIVE_ORTHANT // TODO
};

// Reference value f_ref of the Armijo condition f(x_new) <= f_ref + sigma alpha <grad f(x), eta>
enum class LineSearchReference : std::uint8_t
{
    MONOTONE,   // f(x_k)
    MAX,        // max of the last LineSearchOptions::memory values (Grippo, Lampariello, Lucidi 1986)
    AVERAGE     // weighted average C_k of all values (Zhang, Hager 2004)
};

// Starting step size of the line search (or step size without line search)
enum class InitialStep : std::uint8_t
{
    FIXED,      // LineSearchOptions::alpha (DescentOptions::step_size without line search)
    BB1,        // Barzilai-Borwein <s, s> / <s, y>
    BB2,        // Barzilai-Borwein <s, y> / <y, y>
    ALTERNATING // BB1 and BB2 in turn
};

// ----- Option structures
struct DescentOptions
{
    double tol_lambda;          // tolerance for rayleigh quotients
    double tol_residual;        // tolerance for M-residual
    double step_size;           // fixed step-size used in iteration steps
    unsigned int max_iter;      // maximum GD iterations
    bool line_search;           // determine step-size by line search
    double boundary_fraction = 0.99;  // fraction theta in (0, 1) of ManifoldBase::max_step() taken at most per step

    struct LineSearchOptions
    {
        unsigned int max_iter;    // maximum line search iterations
        double alpha;             // starting step-size for backtracking
        double beta;              // reduction factor for backtracking
        double sigma;             // factor for sufficient decrease
        double min;               // minimum step-size taken

        // Nonmonotone line search and Barzilai-Borwein steps (GradientDescent only; defaults: monotone, fixed start)
        LineSearchReference reference = LineSearchReference::MONOTONE;  // reference value of the Armijo condition
        unsigned int memory = 10; // LineSearchReference::MAX: number of previous values
        double eta = 0.85;        // LineSearchReference::AVERAGE: weight in [0, 1] of the previous average
        InitialStep initial = InitialStep::FIXED;  // starting step size (or fixed step size without line search)
        double bb_min = 1e-10;    // safeguards of the Barzilai-Borwein step size
        double bb_max = 1e10;
    } ls;
};

struct SolverOptions
{
    unsigned int max_inner;     // maximum sparse solver iterations
    double tol_inner;           // relative tolerance for inner solver
    double tol_inner_res;       // relative tolerance for inner solver, w.r.t residual
    SolverMethod solver;        // method for solving sparse linear equations
    Precondition precond;       // preconditioner for solving sparse linear equations
};

struct MG_Options
{
    //bool multilevel;
    unsigned int n_levels;      // number of levels for global refinement
    //unsigned int min_level;     // minimum level for multilevel algorithms
    //unsigned int max_level;     // maximum level for multilevel algorithms
    std::vector<unsigned> v_levels;
};

// Parameters of the FAS cycle itself, which is problem-independent;
// choice of oracles and transfer operators is problem-specific, see e.g. gpe::CoarseModelOptions
struct FAS_Options
{
    double kappa;           // weight for ratio of restricted and coarse gradient
    double eps;             // minimum norm of restricted gradient
    unsigned coarse_every;  // minimum number of fine steps before coarse step is taken
    bool coarse_energy_adaptive;  // solve coarse model with energy-adaptive gradient descent
};

// Output of solutions, independent of the problem
struct OutputOptions
{
    bool output_bin;              // write all iterates in binary format
    std::string bin_filename;     // base name of binary files (empty: default name)
    bool output_vtk;              // write solutions in VTK format
    std::string vtk_filename;     // base name of VTK files (empty: default name)
    unsigned output_every;        // also write every k-th and the final iterate as a VTK series (0: final iterate only)
};


// ---------- Gross-Pitaevskii problem ----------
namespace gpe
{

enum class Potential : std::uint8_t
{
    ZERO,
    CONSTANT,
    SQUARE,
    OPTICAL_LATTICE,
    EXPRESSION      // muparser expression, see GPE_Options::potential_expr
};

// Starting value of the iteration, see ModelBuilder::initial_value()
enum class InitialValue : std::uint8_t
{
    CONSTANT,   // x = value (default 1)
    COSINE,     // cosine bump of radius r (default: domain radius), see CosineBump
    RANDOM      // uniform in [0.5, 1.5]
};

enum class Transport : std::uint8_t
{
    FROBENIUS,            // Version IV  (orth. proj. Frobenius)
    MASS,                 // Version IV  (orth. proj. Mass)
    DIFFERENTIAL,         // Version VI  (Mixed)
    ADJOINT_RESTRICTION,  // Version V
    ADJOINT_DIFFERENTIAL, // Version III
    ADJOINT_RESTRICTION_FROBENIUS,  // Frobenius-metric counterpart of ADJOINT_RESTRICTION, same as FROBENIUS
    ADJOINT_DIFFERENTIAL_FROBENIUS, // Frobenius-metric counterpart of ADJOINT_DIFFERENTIAL
    DIFFERENTIAL_FROBENIUS,         // Frobenius-metric counterpart of DIFFERENTIAL
    // DIFFERENTIAL_MASS,
    // ADJOINT_DIFFERENTIAL_MASS,
};

struct GPE_Options
{
    int dimension;          // dimension of domain
    int degree;             // degree of shape functions
    double radius;          // radius of the cube (square, line) domain
    double beta;            // factor for the non-linear term in GPE
    Ordering order;         // ordering for degrees of freedom
    BoundaryCondition bc;   // problem boundary conditions (dirichlet or neumann)
    MeshKind mesh_kind;     // subdivide the grid into simplices or quadrilaterals
    bool mass_lumping;      // lumped (diagonal) mass matrices, see GrossPitaevskiiLumpedSystem
    Potential potential;    // used potential V for matrix M_V
    std::string potential_expr;  // expression for Potential::EXPRESSION, in the coordinates x[,y[,z]]
    InitialValue initial;   // type of starting point
    ManifoldKind manifold;  // type of manifold
    std::optional<double> initial_arg;  // value (CONSTANT) or radius (COSINE); empty: default
    unsigned seed;          // seed of the random engines, e.g. of initial_value()
};

// Selection of the multilevel components for the GP problem (main_coarse.cc)
// TODO: MetricKind is still defined in rmo (used by MetricBase::kind)
struct CoarseModelOptions
{
    MetricKind metric_t;    // type of coarse oracle (shift metric, gradient metric); NONE: single-level
    MetricKind smooth_t;    // type of fine oracle (gradient descent on fine level)
    MetricKind ccond_t;     // metric for evaluating coarse condition
    Transport transport_t;  // type of vector transport
};

} // namespace gpe

} // namespace rmo

#endif //RMO_OPTION_TYPES_H
